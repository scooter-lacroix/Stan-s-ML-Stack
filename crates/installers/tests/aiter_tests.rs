//! AITER Installer Tests
//!
//! Unit and integration tests for the AMD AI Tensor Engine installer.

use mlstack_installers::{
    AiterBuildConfig, AiterEnvironment, AiterInstaller, AiterSource, AiterVersion, GpuArchitecture,
    Installer,
};
use std::path::PathBuf;

#[test]
fn test_gpu_architecture_values() {
    assert_eq!(GpuArchitecture::Gfx1100.as_str(), "gfx1100");
    assert_eq!(GpuArchitecture::Gfx1101.as_str(), "gfx1101");
    assert_eq!(GpuArchitecture::Gfx1102.as_str(), "gfx1102");
}

#[test]
fn test_gpu_architecture_all_supported() {
    assert_eq!(GpuArchitecture::all_supported(), "gfx1100;gfx1101;gfx1102");
}

#[test]
fn test_gpu_architecture_card_names() {
    let gfx1100_cards = GpuArchitecture::Gfx1100.card_names();
    assert!(gfx1100_cards.contains(&"RX 7900 XTX"));
    assert!(gfx1100_cards.contains(&"Radeon PRO W7900"));
}

#[test]
fn test_aiter_version_creation() {
    let version = AiterVersion::new("0.2.0", "6.0");
    assert_eq!(version.version, "0.2.0");
    assert!(version.git_ref.contains("0.2.0"));
    assert_eq!(version.min_rocm_version, "6.0");
}

#[test]
fn test_aiter_version_latest() {
    let version = AiterVersion::latest();
    assert!(!version.version.is_empty());
    assert!(version.target_archs.len() > 0);
}

#[test]
fn test_aiter_build_config_default() {
    let config = AiterBuildConfig::default();
    assert!(!config.build_isolation); // AITER uses no-build-isolation by default
    assert!(config.install_deps);
    assert!(config.parallel_jobs > 0);
}

#[test]
fn test_aiter_environment_default() {
    let env = AiterEnvironment::default();
    assert_eq!(env.rocm_path, PathBuf::from("/opt/rocm"));
    assert_eq!(env.hsa_gfx_version, "11.0.0");
}

#[test]
fn test_aiter_environment_as_env_map() {
    let env = AiterEnvironment::default();
    let build = AiterBuildConfig::default();
    let map = env.as_env_map(&build);

    assert!(map.contains_key("ROCM_PATH"));
    assert!(map.contains_key("HSA_OVERRIDE_GFX_VERSION"));
    assert!(map.contains_key("PYTORCH_ROCM_ARCH"));
}

#[test]
fn test_installer_latest() {
    let installer = AiterInstaller::latest();
    assert_eq!(installer.name(), "AITER");
}

#[test]
fn test_installer_new() {
    let installer = AiterInstaller::new(AiterVersion::new("0.2.0", "6.0"), AiterSource::RocmGithub);
    assert_eq!(installer.name(), "AITER");
}

#[test]
fn test_installer_with_rocm_path() {
    let installer = AiterInstaller::latest().with_rocm_path("/custom/rocm");
    assert_eq!(installer.name(), "AITER");
}

#[test]
fn test_installer_with_build_config() {
    let config = AiterBuildConfig::default();

    let installer = AiterInstaller::latest().with_build_config(config);
    assert_eq!(installer.name(), "AITER");
}

#[tokio::test]
async fn test_installer_is_installed() {
    let installer = AiterInstaller::latest();
    let result = installer.is_installed().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_installer_preflight_check() {
    let installer = AiterInstaller::latest();
    let result = installer.preflight_check().await;
    assert!(result.is_ok());
}

#[test]
fn test_verification_result() {
    let installer = AiterInstaller::latest();
    let result = installer.verify_installation();
    // Result should be Ok even if not installed
    assert!(result.is_ok());
}
