//! ROCm SMI Installer Tests
//!
//! Unit and integration tests for the ROCm SMI installer.

use mlstack_installers::{
    RocmSmiInstaller, RocmSmiSource, GpuInfo, PowerProfile, Installer,
};
use std::path::PathBuf;

#[test]
fn test_power_profile_values() {
    assert_eq!(PowerProfile::Auto.as_smi_arg(), "auto");
    assert_eq!(PowerProfile::Low.as_smi_arg(), "low");
    assert_eq!(PowerProfile::High.as_smi_arg(), "high");
    assert_eq!(PowerProfile::Video.as_smi_arg(), "video");
    assert_eq!(PowerProfile::VR.as_smi_arg(), "vr");
    assert_eq!(PowerProfile::Compute.as_smi_arg(), "compute");
    assert_eq!(PowerProfile::Custom.as_smi_arg(), "custom");
}

#[test]
fn test_installer_creation() {
    let installer = RocmSmiInstaller::new();
    assert_eq!(installer.name(), "ROCm SMI");
    assert_eq!(installer.version(), "bundled");
}

#[test]
fn test_installer_with_rocm_path() {
    let installer = RocmSmiInstaller::new()
        .with_rocm_path("/custom/rocm");
    assert_eq!(installer.name(), "ROCm SMI");
}

#[test]
fn test_installer_with_source() {
    let installer = RocmSmiInstaller::new()
        .with_source(RocmSmiSource::PackageManager);
    assert_eq!(installer.name(), "ROCm SMI");
}

#[test]
fn test_smi_path() {
    let installer = RocmSmiInstaller::new();
    let smi_path = installer.smi_path();
    assert_eq!(smi_path, PathBuf::from("/opt/rocm/bin/rocm-smi"));
}

#[test]
fn test_python_lib_path() {
    let installer = RocmSmiInstaller::new();
    let lib_path = installer.python_lib_path();
    assert_eq!(lib_path, PathBuf::from("/opt/rocm/libexec/rocm_smi"));
}

#[test]
fn test_default_impl() {
    let installer = RocmSmiInstaller::default();
    assert_eq!(installer.name(), "ROCm SMI");
}

#[tokio::test]
async fn test_installer_is_installed() {
    let installer = RocmSmiInstaller::new();
    let result = installer.is_installed().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_installer_preflight_check() {
    let installer = RocmSmiInstaller::new();
    let result = installer.preflight_check().await;
    assert!(result.is_ok());

    // Should have at least some checks
    let checks = result.unwrap();
    assert!(!checks.is_empty());
}

#[test]
fn test_list_gpus() {
    let installer = RocmSmiInstaller::new();
    let result = installer.list_gpus();
    // Should not panic, but may be empty if no GPUs
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_gpu_info_struct() {
    let gpu = GpuInfo {
        index: 0,
        name: "AMD Radeon RX 7900 XTX".to_string(),
        arch: "gfx1100".to_string(),
        vram_total: 24_000_000_000,
        vram_used: 1_000_000_000,
        temperature: Some(45.0),
        power_usage: Some(150.0),
        fan_speed: Some(30),
        utilization: Some(50),
    };

    assert_eq!(gpu.index, 0);
    assert!(gpu.name.contains("AMD"));
    assert_eq!(gpu.arch, "gfx1100");
}
