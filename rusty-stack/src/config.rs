use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallerConfig {
    pub scripts_dir: String,
    pub log_dir: String,
    pub install_path: String,
    pub batch_mode: bool,
    pub auto_confirm: bool,
    pub star_repos: bool,
    pub force_reinstall: bool,
    pub theme: String,
    pub performance_profile: String,
    #[serde(skip)]
    pub config_path: PathBuf,
}

impl InstallerConfig {
    pub fn load_or_default(scripts_dir: &str) -> Result<Self> {
        let config_path = config_file_path()?;
        let log_dir = config_path
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("logs"))
            .unwrap_or_else(|| PathBuf::from("/tmp/mlstack/logs"));

        let mut config = InstallerConfig::default_with_paths(
            scripts_dir,
            log_dir.to_string_lossy().to_string(),
            config_path.clone(),
        );

        if config_path.exists() {
            let raw = fs::read_to_string(&config_path).context("Failed to read config file")?;
            let value: Value = serde_json::from_str(&raw).unwrap_or_else(|_| json!({}));
            config.apply_from_value(&value);
        } else {
            // Only try to save if we can, but don't crash if we can't
            let _ = config.save(None);
        }

        Ok(config)
    }

    pub fn save(&self, existing: Option<Value>) -> Result<()> {
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent).context("Failed to create config directory")?;
        }
        let mut value = existing.unwrap_or_else(|| json!({}));

        value["scripts_dir"] = json!(self.scripts_dir);
        value["log_dir"] = json!(self.log_dir);
        value["batch_mode"] = json!(self.batch_mode);

        let user_prefs = value
            .get("user_preferences")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let mut user_prefs = user_prefs;
        user_prefs["installation_path"] = json!(self.install_path);
        user_prefs["auto_confirm"] = json!(self.auto_confirm);
        user_prefs["star_repos"] = json!(self.star_repos);
        user_prefs["force_reinstall"] = json!(self.force_reinstall);
        user_prefs["theme"] = json!(self.theme);
        user_prefs["performance_profile"] = json!(self.performance_profile);
        value["user_preferences"] = user_prefs;

        fs::write(&self.config_path, serde_json::to_string_pretty(&value)?)
            .context("Failed to save config")?;

        Ok(())
    }

    fn apply_from_value(&mut self, value: &Value) {
        if let Some(dir) = value.get("scripts_dir").and_then(|v| v.as_str()) {
            self.scripts_dir = dir.to_string();
        }
        if let Some(dir) = value.get("log_dir").and_then(|v| v.as_str()) {
            self.log_dir = dir.to_string();
        }
        if let Some(batch) = value.get("batch_mode").and_then(|v| v.as_bool()) {
            self.batch_mode = batch;
        }

        if let Some(prefs) = value.get("user_preferences") {
            if let Some(path) = prefs.get("installation_path").and_then(|v| v.as_str()) {
                self.install_path = path.to_string();
            }
            if let Some(auto) = prefs.get("auto_confirm").and_then(|v| v.as_bool()) {
                self.auto_confirm = auto;
            }
            if let Some(star) = prefs.get("star_repos").and_then(|v| v.as_bool()) {
                self.star_repos = star;
            }
            if let Some(force) = prefs.get("force_reinstall").and_then(|v| v.as_bool()) {
                self.force_reinstall = force;
            }
            if let Some(theme) = prefs.get("theme").and_then(|v| v.as_str()) {
                self.theme = theme.to_string();
            }
            if let Some(profile) = prefs.get("performance_profile").and_then(|v| v.as_str()) {
                self.performance_profile = profile.to_string();
            }
        }
    }

    pub fn default_with_paths(scripts_dir: &str, log_dir: String, config_path: PathBuf) -> Self {
        Self {
            scripts_dir: scripts_dir.to_string(),
            log_dir,
            install_path: "/opt/rocm".into(),
            batch_mode: std::env::var("MLSTACK_BATCH_MODE").unwrap_or_default() == "1",
            auto_confirm: false,
            star_repos: true,
            force_reinstall: false,
            theme: "dark".into(),
            performance_profile: "balanced".into(),
            config_path,
        }
    }
}

pub fn config_file_path() -> Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let config_dir = PathBuf::from(home).join(".mlstack").join("config");
    fs::create_dir_all(&config_dir).context("Failed to create config directory")?;
    Ok(config_dir.join("config.json"))
}
