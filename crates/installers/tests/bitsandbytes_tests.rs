//! BitsAndBytes Installer Tests
//!
//! Unit and integration tests for the BitsAndBytes installer.

use mlstack_installers::{
    BitsAndBytesInstaller, BitsAndBytesVersion,
    BitsAndBytesBuildConfig, QuantizationBits, Installer,
};

#[test]
fn test_quantization_bits_memory_factor() {
    assert_eq!(QuantizationBits::Int8.memory_factor(), 2.0);
    assert_eq!(QuantizationBits::Int4.memory_factor(), 4.0);
    assert_eq!(QuantizationBits::Fp4.memory_factor(), 4.0);
    assert_eq!(QuantizationBits::Nf4.memory_factor(), 4.0);
}

#[test]
fn test_quantization_bits_transformers_key() {
    assert_eq!(QuantizationBits::Int8.transformers_key(), "load_in_8bit");
    assert_eq!(QuantizationBits::Int4.transformers_key(), "load_in_4bit");
    assert_eq!(QuantizationBits::Fp4.transformers_key(), "load_in_4bit");
    assert_eq!(QuantizationBits::Nf4.transformers_key(), "load_in_4bit");
}

#[test]
fn test_version_creation() {
    let version = BitsAndBytesVersion::new("0.45.0");
    assert_eq!(version.version, "0.45.0");
    assert!(version.rocm_native);
}

#[test]
fn test_version_latest() {
    let version = BitsAndBytesVersion::latest_stable();
    assert!(!version.version.is_empty());
    assert!(version.rocm_native);
}

#[test]
fn test_build_config_default() {
    let config = BitsAndBytesBuildConfig::default();
    assert!(config.use_hip);
    assert!(config.cuda_compat);
    assert!(config.gpu_archs.contains(&"gfx1100".to_string()));
    assert!(config.parallel_jobs > 0);
}

#[test]
fn test_installer_from_pypi() {
    let installer = BitsAndBytesInstaller::from_pypi();
    assert_eq!(installer.name(), "BitsAndBytes");
}

#[test]
fn test_installer_from_github() {
    let installer = BitsAndBytesInstaller::from_github();
    assert_eq!(installer.name(), "BitsAndBytes");
}

#[test]
fn test_installer_from_rocm_fork() {
    let installer = BitsAndBytesInstaller::from_rocm_fork();
    assert_eq!(installer.name(), "BitsAndBytes");
}

#[test]
fn test_installer_with_rocm_path() {
    let installer = BitsAndBytesInstaller::from_pypi()
        .with_rocm_path("/custom/rocm");
    assert_eq!(installer.name(), "BitsAndBytes");
}

#[test]
fn test_installer_with_build_config() {
    let mut config = BitsAndBytesBuildConfig::default();
    config.cuda_compat = false;

    let installer = BitsAndBytesInstaller::from_pypi()
        .with_build_config(config);
    assert_eq!(installer.name(), "BitsAndBytes");
}

#[tokio::test]
async fn test_installer_is_installed() {
    let installer = BitsAndBytesInstaller::from_pypi();
    let result = installer.is_installed().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_installer_preflight_check() {
    let installer = BitsAndBytesInstaller::from_pypi();
    let result = installer.preflight_check().await;
    assert!(result.is_ok());
}

#[test]
fn test_verification_result() {
    let installer = BitsAndBytesInstaller::from_pypi();
    let result = installer.verify_installation();
    assert!(result.is_ok());
}
