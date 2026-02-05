//! DeepSpeed Installer Tests
//!
//! Unit and integration tests for the DeepSpeed installer.

use mlstack_installers::{
    generate_ds_config, DeepSpeedBuildConfig, DeepSpeedEnvironment, DeepSpeedInstaller,
    DeepSpeedOps, DeepSpeedVersion, Installer, ZeroStage,
};
use std::path::PathBuf;

#[test]
fn test_zero_stage_values() {
    assert_eq!(ZeroStage::Disabled as u8, 0);
    assert_eq!(ZeroStage::Stage1 as u8, 1);
    assert_eq!(ZeroStage::Stage2 as u8, 2);
    assert_eq!(ZeroStage::Stage3 as u8, 3);
}

#[test]
fn test_zero_stage_as_int() {
    assert_eq!(ZeroStage::Disabled.as_int(), 0);
    assert_eq!(ZeroStage::Stage1.as_int(), 1);
    assert_eq!(ZeroStage::Stage2.as_int(), 2);
    assert_eq!(ZeroStage::Stage3.as_int(), 3);
}

#[test]
fn test_zero_stage_memory_savings() {
    assert!(ZeroStage::Stage3.memory_savings().len() > 0);
    assert!(ZeroStage::Stage2.memory_savings().len() > 0);
    assert!(ZeroStage::Stage1.memory_savings().len() > 0);
    assert!(ZeroStage::Disabled.memory_savings() == "None");
}

#[test]
fn test_version_creation() {
    let version = DeepSpeedVersion::new("0.16.0");
    assert_eq!(version.version, "0.16.0");
    assert!(version.git_ref.contains("0.16.0"));
}

#[test]
fn test_version_latest() {
    let version = DeepSpeedVersion::latest();
    assert!(!version.version.is_empty());
}

#[test]
fn test_build_config_default() {
    let config = DeepSpeedBuildConfig::default();
    // Check ops field which is a DeepSpeedOps
    assert!(config.parallel_jobs > 0);
}

#[test]
fn test_ops_default() {
    let ops = DeepSpeedOps::default();
    let env = ops.as_build_env();
    // Default should have some ops enabled
    assert!(!env.is_empty());
}

#[test]
fn test_environment_default() {
    let env = DeepSpeedEnvironment::default();
    assert_eq!(env.rocm_path, PathBuf::from("/opt/rocm"));
}

#[test]
fn test_environment_as_env_map() {
    let env = DeepSpeedEnvironment::default();
    let config = DeepSpeedBuildConfig::default();
    let map = env.as_env_map(&config);

    assert!(map.contains_key("ROCM_PATH"));
}

#[test]
fn test_generate_ds_config_disabled() {
    let config = generate_ds_config(ZeroStage::Disabled, 16, 1, true);

    let zero = config.get("zero_optimization").unwrap();
    assert_eq!(zero.get("stage").unwrap().as_u64().unwrap(), 0);
}

#[test]
fn test_generate_ds_config_stage2() {
    let config = generate_ds_config(ZeroStage::Stage2, 16, 1, true);

    let zero = config.get("zero_optimization").unwrap();
    assert_eq!(zero.get("stage").unwrap().as_u64().unwrap(), 2);
    assert!(zero.get("contiguous_gradients").is_some());
}

#[test]
fn test_generate_ds_config_stage3() {
    let config = generate_ds_config(ZeroStage::Stage3, 16, 1, true);

    let zero = config.get("zero_optimization").unwrap();
    assert_eq!(zero.get("stage").unwrap().as_u64().unwrap(), 3);
    assert!(zero.get("offload_optimizer").is_some());
    assert!(zero.get("offload_param").is_some());
}

#[test]
fn test_generate_ds_config_has_fp16() {
    let config = generate_ds_config(ZeroStage::Stage2, 16, 1, true);
    assert!(config.get("fp16").is_some());
}

#[test]
fn test_installer_from_pypi() {
    let installer = DeepSpeedInstaller::from_pypi();
    assert_eq!(installer.name(), "DeepSpeed");
}

#[test]
fn test_installer_from_github() {
    let installer = DeepSpeedInstaller::from_github();
    assert_eq!(installer.name(), "DeepSpeed");
}

#[test]
fn test_installer_with_rocm_path() {
    let installer = DeepSpeedInstaller::from_pypi().with_rocm_path("/custom/rocm");
    assert_eq!(installer.name(), "DeepSpeed");
}

#[test]
fn test_installer_with_ops() {
    let installer = DeepSpeedInstaller::from_pypi().with_ops(DeepSpeedOps::default());
    assert_eq!(installer.name(), "DeepSpeed");
}

#[tokio::test]
async fn test_installer_is_installed() {
    let installer = DeepSpeedInstaller::from_pypi();
    let result = installer.is_installed().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_installer_preflight_check() {
    let installer = DeepSpeedInstaller::from_pypi();
    let result = installer.preflight_check().await;
    assert!(result.is_ok());
}

#[test]
fn test_verification_result() {
    let installer = DeepSpeedInstaller::from_pypi();
    let result = installer.verify_installation();
    assert!(result.is_ok());
}
