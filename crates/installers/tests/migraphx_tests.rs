//! MIGraphX Installer Tests
//!
//! Unit and integration tests for the AMD MIGraphX installer.

use mlstack_installers::{
    MigraphxInstaller, MigraphxSource, MigraphxVersion, MigraphxBuildConfig,
    MigraphxEnvironment, OptimizationTarget, ModelFormat, Installer,
};
use std::path::PathBuf;

#[test]
fn test_optimization_target_default() {
    let target = OptimizationTarget::default();
    assert!(matches!(target, OptimizationTarget::Inference));
}

#[test]
fn test_model_format_values() {
    // Ensure all formats are valid
    let _onnx = ModelFormat::Onnx;
    let _tf = ModelFormat::TensorFlow;
    let _mgx = ModelFormat::Migraphx;
}

#[test]
fn test_source_default() {
    let source = MigraphxSource::default();
    assert!(matches!(source, MigraphxSource::PackageManager));
}

#[test]
fn test_version_creation() {
    let version = MigraphxVersion::new("6.4");
    assert_eq!(version.version, "6.4");
    assert!(version.git_ref.contains("rocm"));
}

#[test]
fn test_version_latest() {
    let version = MigraphxVersion::latest();
    assert!(!version.version.is_empty());
}

#[test]
fn test_build_config_default() {
    let config = MigraphxBuildConfig::default();
    assert!(config.python_bindings);
    assert!(config.onnx_support);
    assert!(config.gpu_archs.contains(&"gfx1100".to_string()));
    assert!(config.parallel_jobs > 0);
    assert_eq!(config.build_type, "Release");
}

#[test]
fn test_environment_default() {
    let env = MigraphxEnvironment::default();
    assert_eq!(env.rocm_path, PathBuf::from("/opt/rocm"));
    assert_eq!(env.hsa_gfx_version, "11.0.0");
}

#[test]
fn test_environment_as_env_map() {
    let env = MigraphxEnvironment::default();
    let map = env.as_env_map();

    assert!(map.contains_key("ROCM_PATH"));
    assert!(map.contains_key("HSA_OVERRIDE_GFX_VERSION"));
    assert!(map.contains_key("MIGRAPHX_DISABLE_FAST_GELU"));
}

#[test]
fn test_installer_from_package_manager() {
    let installer = MigraphxInstaller::from_package_manager();
    assert_eq!(installer.name(), "MIGraphX");
}

#[test]
fn test_installer_from_source() {
    let installer = MigraphxInstaller::from_source();
    assert_eq!(installer.name(), "MIGraphX");
}

#[test]
fn test_installer_with_rocm_path() {
    let installer = MigraphxInstaller::from_package_manager()
        .with_rocm_path("/custom/rocm");
    assert_eq!(installer.name(), "MIGraphX");
}

#[test]
fn test_installer_with_build_config() {
    let mut config = MigraphxBuildConfig::default();
    config.tensorflow_support = true;

    let installer = MigraphxInstaller::from_source()
        .with_build_config(config);
    assert_eq!(installer.name(), "MIGraphX");
}

#[tokio::test]
async fn test_installer_is_installed() {
    let installer = MigraphxInstaller::from_package_manager();
    let result = installer.is_installed().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_installer_preflight_check() {
    let installer = MigraphxInstaller::from_package_manager();
    let result = installer.preflight_check().await;
    assert!(result.is_ok());
}

#[test]
fn test_verification_result() {
    let installer = MigraphxInstaller::from_package_manager();
    let result = installer.verify_installation();
    assert!(result.is_ok());
}
