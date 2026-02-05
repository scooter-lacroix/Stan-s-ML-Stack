//! Weights & Biases Installer Tests
//!
//! Unit and integration tests for the W&B installer.

use mlstack_installers::{
    WandbInstaller, WandbVersion, WandbConfig, WandbMode, Installer,
};
use std::path::PathBuf;

#[test]
fn test_wandb_mode_values() {
    assert_eq!(WandbMode::Online.as_env_value(), "online");
    assert_eq!(WandbMode::Offline.as_env_value(), "offline");
    assert_eq!(WandbMode::Disabled.as_env_value(), "disabled");
    assert_eq!(WandbMode::DryRun.as_env_value(), "dryrun");
}

#[test]
fn test_wandb_mode_default() {
    let mode = WandbMode::default();
    assert!(matches!(mode, WandbMode::Online));
}

#[test]
fn test_wandb_config_default() {
    let config = WandbConfig::default();
    assert!(matches!(config.mode, WandbMode::Online));
    assert!(config.project.is_none());
    assert!(config.entity.is_none());
    assert!(config.name.is_none());
    assert!(config.tags.is_empty());
    assert!(config.notes.is_none());
    assert_eq!(config.dir, PathBuf::from("./wandb"));
    assert_eq!(config.log_freq, 100);
    assert!(config.save_code);
    assert!(!config.watch_gradients);
}

#[test]
fn test_wandb_config_generate_init_code() {
    let config = WandbConfig::default();
    let code = config.generate_init_code();

    assert!(code.contains("import wandb"));
    assert!(code.contains("wandb.init"));
    assert!(code.contains("project="));
    assert!(code.contains("mode="));
    assert!(code.contains("save_code="));
}

#[test]
fn test_wandb_config_with_project() {
    let mut config = WandbConfig::default();
    config.project = Some("my-project".to_string());

    let code = config.generate_init_code();
    assert!(code.contains("\"my-project\""));
}

#[test]
fn test_wandb_config_with_tags() {
    let mut config = WandbConfig::default();
    config.tags = vec!["test".to_string(), "experiment".to_string()];

    let code = config.generate_init_code();
    assert!(code.contains("\"test\""));
    assert!(code.contains("\"experiment\""));
}

#[test]
fn test_wandb_config_as_env_map() {
    let mut config = WandbConfig::default();
    config.project = Some("test-project".to_string());
    config.entity = Some("test-entity".to_string());
    config.mode = WandbMode::Offline;
    config.tags = vec!["tag1".to_string(), "tag2".to_string()];

    let env = config.as_env_map();

    assert_eq!(env.get("WANDB_MODE"), Some(&"offline".to_string()));
    assert_eq!(env.get("WANDB_PROJECT"), Some(&"test-project".to_string()));
    assert_eq!(env.get("WANDB_ENTITY"), Some(&"test-entity".to_string()));
    assert_eq!(env.get("WANDB_TAGS"), Some(&"tag1,tag2".to_string()));
    assert!(env.contains_key("WANDB_DIR"));
}

#[test]
fn test_wandb_config_env_map_minimal() {
    let config = WandbConfig::default();
    let env = config.as_env_map();

    // Should always have mode and dir
    assert!(env.contains_key("WANDB_MODE"));
    assert!(env.contains_key("WANDB_DIR"));

    // Should not have optional fields
    assert!(!env.contains_key("WANDB_PROJECT"));
    assert!(!env.contains_key("WANDB_ENTITY"));
    assert!(!env.contains_key("WANDB_TAGS"));
}

#[test]
fn test_wandb_version_creation() {
    let version = WandbVersion::new("0.19.0");
    assert_eq!(version.version, "0.19.0");
}

#[test]
fn test_wandb_version_latest() {
    let version = WandbVersion::latest();
    assert!(!version.version.is_empty());
}

#[test]
fn test_wandb_version_with_extras() {
    let version = WandbVersion::latest()
        .with_extras(vec!["sweeps".to_string(), "launch".to_string()]);

    assert_eq!(version.extras.len(), 2);
    assert!(version.extras.contains(&"sweeps".to_string()));
}

#[test]
fn test_installer_latest() {
    let installer = WandbInstaller::latest();
    assert_eq!(installer.name(), "Weights & Biases");
}

#[test]
fn test_installer_with_project() {
    let installer = WandbInstaller::latest()
        .with_project("my-project");
    assert_eq!(installer.name(), "Weights & Biases");
}

#[test]
fn test_installer_with_entity() {
    let installer = WandbInstaller::latest()
        .with_entity("my-entity");
    assert_eq!(installer.name(), "Weights & Biases");
}

#[test]
fn test_installer_with_mode() {
    let installer = WandbInstaller::latest()
        .with_mode(WandbMode::Offline);
    assert_eq!(installer.name(), "Weights & Biases");
}

#[test]
fn test_installer_with_config() {
    let config = WandbConfig::default();
    let installer = WandbInstaller::latest()
        .with_config(config);
    assert_eq!(installer.name(), "Weights & Biases");
}

#[tokio::test]
async fn test_installer_is_installed() {
    let installer = WandbInstaller::latest();
    let result = installer.is_installed().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_installer_preflight_check() {
    let installer = WandbInstaller::latest();
    let result = installer.preflight_check().await;
    assert!(result.is_ok());
}

#[test]
fn test_verification_result() {
    let installer = WandbInstaller::latest();
    let result = installer.verify_installation();
    assert!(result.is_ok());
}

#[test]
fn test_is_logged_in() {
    let installer = WandbInstaller::latest();
    // Should not panic, regardless of login status
    let _ = installer.is_logged_in();
}
