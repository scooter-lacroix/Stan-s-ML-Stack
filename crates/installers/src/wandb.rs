//! Weights & Biases (wandb) Installer
//!
//! Pure Rust implementation for installing and configuring Weights & Biases
//! for ML experiment tracking with AMD GPU support.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

/// W&B run mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WandbMode {
    /// Online mode (sync to cloud)
    #[default]
    Online,
    /// Offline mode (local only)
    Offline,
    /// Disabled
    Disabled,
    /// Dry run (no logging)
    DryRun,
}

impl WandbMode {
    /// Returns environment variable value.
    pub fn as_env_value(&self) -> &'static str {
        match self {
            WandbMode::Online => "online",
            WandbMode::Offline => "offline",
            WandbMode::Disabled => "disabled",
            WandbMode::DryRun => "dryrun",
        }
    }
}

/// W&B configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WandbConfig {
    /// Run mode
    pub mode: WandbMode,
    /// Project name
    pub project: Option<String>,
    /// Entity (user or team)
    pub entity: Option<String>,
    /// Run name
    pub name: Option<String>,
    /// Run tags
    pub tags: Vec<String>,
    /// Run notes
    pub notes: Option<String>,
    /// Log directory
    pub dir: PathBuf,
    /// Log frequency (steps)
    pub log_freq: usize,
    /// Save code
    pub save_code: bool,
    /// Watch gradients
    pub watch_gradients: bool,
}

impl Default for WandbConfig {
    fn default() -> Self {
        Self {
            mode: WandbMode::Online,
            project: None,
            entity: None,
            name: None,
            tags: Vec::new(),
            notes: None,
            dir: PathBuf::from("./wandb"),
            log_freq: 100,
            save_code: true,
            watch_gradients: false,
        }
    }
}

impl WandbConfig {
    /// Generates wandb.init() code snippet.
    pub fn generate_init_code(&self) -> String {
        let tags_str = if self.tags.is_empty() {
            "None".to_string()
        } else {
            format!("[{}]", self.tags.iter().map(|t| format!("\"{}\"", t)).collect::<Vec<_>>().join(", "))
        };

        let project_str = self.project.as_ref().map(|p| format!("\"{}\"", p)).unwrap_or_else(|| "None".to_string());
        let entity_str = self.entity.as_ref().map(|e| format!("\"{}\"", e)).unwrap_or_else(|| "None".to_string());
        let name_str = self.name.as_ref().map(|n| format!("\"{}\"", n)).unwrap_or_else(|| "None".to_string());
        let notes_str = self.notes.as_ref().map(|n| format!("\"{}\"", n)).unwrap_or_else(|| "None".to_string());

        format!(r#"
import wandb

# Initialize W&B run
run = wandb.init(
    project={project},
    entity={entity},
    name={name},
    tags={tags},
    notes={notes},
    dir="{dir}",
    mode="{mode}",
    save_code={save_code},
)

# Optional: Watch model gradients
# wandb.watch(model, log="all", log_freq={log_freq})

# Log metrics
# wandb.log({{"loss": loss, "accuracy": acc}})

# Finish run
# wandb.finish()
"#,
            project = project_str,
            entity = entity_str,
            name = name_str,
            tags = tags_str,
            notes = notes_str,
            dir = self.dir.display(),
            mode = self.mode.as_env_value(),
            save_code = if self.save_code { "True" } else { "False" },
            log_freq = self.log_freq,
        )
    }

    /// Returns environment variables for this config.
    pub fn as_env_map(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        env.insert("WANDB_MODE".to_string(), self.mode.as_env_value().to_string());
        env.insert("WANDB_DIR".to_string(), self.dir.to_string_lossy().to_string());

        if let Some(ref project) = self.project {
            env.insert("WANDB_PROJECT".to_string(), project.clone());
        }

        if let Some(ref entity) = self.entity {
            env.insert("WANDB_ENTITY".to_string(), entity.clone());
        }

        if let Some(ref name) = self.name {
            env.insert("WANDB_RUN_NAME".to_string(), name.clone());
        }

        if !self.tags.is_empty() {
            env.insert("WANDB_TAGS".to_string(), self.tags.join(","));
        }

        if let Some(ref notes) = self.notes {
            env.insert("WANDB_NOTES".to_string(), notes.clone());
        }

        env
    }
}

/// W&B version configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WandbVersion {
    /// Version string
    pub version: String,
    /// Install extras (e.g., "sweeps", "launch")
    pub extras: Vec<String>,
}

impl WandbVersion {
    /// Creates a new version.
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            extras: Vec::new(),
        }
    }

    /// Latest stable version.
    pub fn latest() -> Self {
        Self {
            version: "0.19.9".to_string(),
            extras: Vec::new(),
        }
    }

    /// With specific extras.
    pub fn with_extras(mut self, extras: Vec<String>) -> Self {
        self.extras = extras;
        self
    }
}

/// W&B installer.
pub struct WandbInstaller {
    version: WandbVersion,
    config: WandbConfig,
}

impl WandbInstaller {
    /// Creates a new installer.
    pub fn new(version: WandbVersion) -> Self {
        Self {
            version,
            config: WandbConfig::default(),
        }
    }

    /// Creates installer for latest version.
    pub fn latest() -> Self {
        Self::new(WandbVersion::latest())
    }

    /// Sets configuration.
    pub fn with_config(mut self, config: WandbConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets project name.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.config.project = Some(project.into());
        self
    }

    /// Sets entity.
    pub fn with_entity(mut self, entity: impl Into<String>) -> Self {
        self.config.entity = Some(entity.into());
        self
    }

    /// Sets mode.
    pub fn with_mode(mut self, mode: WandbMode) -> Self {
        self.config.mode = mode;
        self
    }

    /// Checks if wandb is installed.
    fn check_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import wandb; print(wandb.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Gets installed version.
    pub fn get_installed_version(&self) -> Option<String> {
        Command::new("python3")
            .args(["-c", "import wandb; print(wandb.__version__)"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Checks if logged in.
    pub fn is_logged_in(&self) -> bool {
        Command::new("wandb")
            .args(["status"])
            .output()
            .map(|o| {
                let stdout = String::from_utf8_lossy(&o.stdout);
                stdout.contains("Logged in") || stdout.contains("logged in")
            })
            .unwrap_or(false)
    }

    /// Logs in with API key.
    pub fn login(&self, api_key: &str) -> Result<()> {
        let output = Command::new("wandb")
            .args(["login", api_key])
            .output()
            .context("Failed to run wandb login")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Login failed: {}", stderr)
            ).into());
        }

        Ok(())
    }

    /// Logs in with environment variable.
    pub fn login_from_env(&self) -> Result<()> {
        let output = Command::new("wandb")
            .args(["login"])
            .env("WANDB_API_KEY", std::env::var("WANDB_API_KEY").unwrap_or_default())
            .output()
            .context("Failed to run wandb login")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Login failed: {}", stderr)
            ).into());
        }

        Ok(())
    }

    /// Creates a sweep configuration.
    pub fn create_sweep_config(
        &self,
        method: &str,
        metric_name: &str,
        metric_goal: &str,
        parameters: HashMap<String, serde_json::Value>,
    ) -> serde_json::Value {
        serde_json::json!({
            "method": method,
            "metric": {
                "name": metric_name,
                "goal": metric_goal
            },
            "parameters": parameters
        })
    }

    /// Runs verification.
    pub fn verify_installation(&self) -> Result<WandbVerification> {
        let installed = self.check_installed();
        let mut result = WandbVerification {
            installed,
            ..Default::default()
        };

        if !result.installed {
            return Ok(result);
        }

        result.version = self.get_installed_version();
        result.logged_in = self.is_logged_in();

        // Check CLI
        let cli_check = Command::new("wandb").arg("--version").output();
        result.cli_available = cli_check.map(|o| o.status.success()).unwrap_or(false);

        // Check GPU logging capability
        let gpu_check = Command::new("python3")
            .args(["-c", r#"
import torch
import wandb
print(f"GPU_AVAILABLE:{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU_NAME:{torch.cuda.get_device_name(0)}")
"#])
            .output();

        if let Ok(output) = gpu_check {
            let stdout = String::from_utf8_lossy(&output.stdout);
            result.gpu_logging = stdout.contains("GPU_AVAILABLE:True");
            for line in stdout.lines() {
                if line.starts_with("GPU_NAME:") {
                    result.gpu_name = Some(line.strip_prefix("GPU_NAME:").unwrap().to_string());
                }
            }
        }

        Ok(result)
    }
}

/// W&B verification result.
#[derive(Debug, Default)]
pub struct WandbVerification {
    /// Is installed
    pub installed: bool,
    /// Version
    pub version: Option<String>,
    /// CLI available
    pub cli_available: bool,
    /// Logged in
    pub logged_in: bool,
    /// GPU logging available
    pub gpu_logging: bool,
    /// GPU name
    pub gpu_name: Option<String>,
}

#[async_trait::async_trait]
impl Installer for WandbInstaller {
    fn name(&self) -> &str {
        "Weights & Biases"
    }

    fn version(&self) -> &str {
        &self.version.version
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.check_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        if Command::new("python3").arg("--version").output().is_err() {
            checks.push("Python 3 is not installed".to_string());
        }

        if self.check_installed() {
            if let Some(version) = self.get_installed_version() {
                checks.push(format!("wandb {} already installed", version));
            }

            if self.is_logged_in() {
                checks.push("Logged in to W&B".to_string());
            } else {
                checks.push("Not logged in to W&B".to_string());
            }
        }

        // Check for API key in environment
        if std::env::var("WANDB_API_KEY").is_ok() {
            checks.push("WANDB_API_KEY found in environment".to_string());
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Installing Weights & Biases...".to_string());
        }

        // Build package string with extras
        let package = if self.version.extras.is_empty() {
            format!("wandb=={}", self.version.version)
        } else {
            format!("wandb[{}]=={}", self.version.extras.join(","), self.version.version)
        };

        if let Some(ref cb) = progress {
            cb(0.3, format!("Installing {}...", package));
        }

        let output = Command::new("python3")
            .args(["-m", "pip", "install", &package])
            .output()
            .context("Failed to install wandb")?;

        if !output.status.success() {
            // Try without version constraint
            let output = Command::new("python3")
                .args(["-m", "pip", "install", "wandb"])
                .output()
                .context("Failed to install wandb")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(InstallerError::InstallationFailed(
                    format!("Installation failed: {}", stderr)
                ).into());
            }
        }

        if let Some(ref cb) = progress {
            cb(0.8, "Installation complete".to_string());
        }

        // Create default directories
        std::fs::create_dir_all(&self.config.dir).ok();

        if let Some(ref cb) = progress {
            cb(1.0, "Weights & Biases installation complete".to_string());
        }

        Ok(())
    }

    async fn uninstall(&self) -> Result<()> {
        let output = Command::new("python3")
            .args(["-m", "pip", "uninstall", "-y", "wandb"])
            .output()
            .context("Failed to uninstall wandb")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Uninstall failed: {}", stderr)
            ).into());
        }

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        let result = self.verify_installation()?;
        Ok(result.installed && result.cli_available)
    }
}

/// Utility function to log AMD GPU metrics to W&B.
pub fn log_amd_gpu_metrics() -> Result<HashMap<String, f64>> {
    let output = Command::new("rocm-smi")
        .args(["--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--json"])
        .output()
        .context("Failed to run rocm-smi")?;

    let mut metrics = HashMap::new();

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
            if let Some(obj) = json.as_object() {
                for (key, value) in obj {
                    if key.starts_with("card") {
                        if let Some(gpu_use) = value.get("GPU use (%)").and_then(|v| v.as_f64()) {
                            metrics.insert(format!("{}_utilization", key), gpu_use);
                        }
                        if let Some(temp) = value.get("Temperature (Sensor edge) (C)").and_then(|v| v.as_f64()) {
                            metrics.insert(format!("{}_temperature", key), temp);
                        }
                        if let Some(power) = value.get("Average Graphics Package Power (W)").and_then(|v| v.as_f64()) {
                            metrics.insert(format!("{}_power", key), power);
                        }
                    }
                }
            }
        }
    }

    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wandb_mode() {
        assert_eq!(WandbMode::Online.as_env_value(), "online");
        assert_eq!(WandbMode::Offline.as_env_value(), "offline");
    }

    #[test]
    fn test_wandb_config_default() {
        let config = WandbConfig::default();
        assert!(config.save_code);
        assert_eq!(config.mode, WandbMode::Online);
    }

    #[test]
    fn test_generate_init_code() {
        let config = WandbConfig::default();
        let code = config.generate_init_code();
        assert!(code.contains("wandb.init"));
        assert!(code.contains("mode="));
    }

    #[test]
    fn test_config_env_map() {
        let mut config = WandbConfig::default();
        config.project = Some("test-project".to_string());
        let env = config.as_env_map();
        assert_eq!(env.get("WANDB_PROJECT"), Some(&"test-project".to_string()));
    }

    #[test]
    fn test_installer_creation() {
        let installer = WandbInstaller::latest();
        assert_eq!(installer.name(), "Weights & Biases");
    }

    #[test]
    fn test_installer_builder() {
        let installer = WandbInstaller::latest()
            .with_project("my-project")
            .with_mode(WandbMode::Offline);

        assert_eq!(installer.config.project, Some("my-project".to_string()));
        assert_eq!(installer.config.mode, WandbMode::Offline);
    }
}
