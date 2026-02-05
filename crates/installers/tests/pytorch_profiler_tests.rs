//! PyTorch Profiler Installer Tests
//!
//! Unit and integration tests for the PyTorch Profiler installer.

use mlstack_installers::{
    PytorchProfilerInstaller, ProfilerConfig, ProfilerEnvironment,
    ProfileActivity, ProfilerOutput, Installer,
};
use std::path::PathBuf;

#[test]
fn test_profile_activity_values() {
    let cpu = ProfileActivity::Cpu;
    let cuda = ProfileActivity::Cuda;
    let both = ProfileActivity::Both;

    assert_eq!(cpu.as_pytorch_activities().len(), 1);
    assert_eq!(cuda.as_pytorch_activities().len(), 1);
    assert_eq!(both.as_pytorch_activities().len(), 2);
}

#[test]
fn test_profile_activity_pytorch_values() {
    let cpu = ProfileActivity::Cpu;
    let activities = cpu.as_pytorch_activities();
    assert!(activities.contains(&"ProfilerActivity.CPU"));

    let cuda = ProfileActivity::Cuda;
    let activities = cuda.as_pytorch_activities();
    assert!(activities.contains(&"ProfilerActivity.CUDA"));
}

#[test]
fn test_profiler_output_default() {
    let output = ProfilerOutput::default();
    assert!(matches!(output, ProfilerOutput::ChromeTrace));
}

#[test]
fn test_profiler_config_default() {
    let config = ProfilerConfig::default();
    assert!(matches!(config.activities, ProfileActivity::Both));
    assert!(config.record_shapes);
    assert!(config.profile_memory);
    assert!(!config.with_stack);
    assert!(config.with_flops);
    assert!(config.with_modules);
    assert_eq!(config.schedule_wait, 1);
    assert_eq!(config.schedule_warmup, 1);
    assert_eq!(config.schedule_active, 3);
    assert_eq!(config.schedule_repeat, 1);
}

#[test]
fn test_profiler_config_generate_code() {
    let config = ProfilerConfig::default();
    let code = config.generate_profiler_code();

    assert!(code.contains("import torch"));
    assert!(code.contains("from torch.profiler import profile"));
    assert!(code.contains("ProfilerActivity"));
    assert!(code.contains("schedule"));
    assert!(code.contains("tensorboard_trace_handler"));
    assert!(code.contains("record_shapes"));
    assert!(code.contains("profile_memory"));
}

#[test]
fn test_profiler_config_custom() {
    let mut config = ProfilerConfig::default();
    config.activities = ProfileActivity::Cpu;
    config.with_stack = true;
    config.schedule_active = 10;

    let code = config.generate_profiler_code();
    assert!(code.contains("\"with_stack\": True"));
    assert!(code.contains("active=10"));
}

#[test]
fn test_profiler_environment_default() {
    let env = ProfilerEnvironment::default();
    assert_eq!(env.rocm_path, PathBuf::from("/opt/rocm"));
    assert!(env.enable_rocm_profiler);
}

#[test]
fn test_profiler_environment_as_env_map() {
    let env = ProfilerEnvironment::default();
    let map = env.as_env_map();

    assert!(map.contains_key("ROCM_PATH"));
}

#[test]
fn test_profiler_environment_disabled() {
    let mut env = ProfilerEnvironment::default();
    env.enable_rocm_profiler = false;
    env.hsa_tools_lib = None;

    let map = env.as_env_map();
    // When disabled and no HSA lib, should set to "0"
    assert_eq!(map.get("HSA_TOOLS_LIB"), Some(&"0".to_string()));
}

#[test]
fn test_installer_creation() {
    let installer = PytorchProfilerInstaller::new();
    assert_eq!(installer.name(), "PyTorch Profiler");
    assert_eq!(installer.version(), "bundled");
}

#[test]
fn test_installer_default() {
    let installer = PytorchProfilerInstaller::default();
    assert_eq!(installer.name(), "PyTorch Profiler");
}

#[test]
fn test_installer_with_rocm_path() {
    let installer = PytorchProfilerInstaller::new()
        .with_rocm_path("/custom/rocm");
    assert_eq!(installer.name(), "PyTorch Profiler");
}

#[test]
fn test_installer_with_config() {
    let config = ProfilerConfig::default();
    let installer = PytorchProfilerInstaller::new()
        .with_config(config);
    assert_eq!(installer.name(), "PyTorch Profiler");
}

#[tokio::test]
async fn test_installer_is_installed() {
    let installer = PytorchProfilerInstaller::new();
    let result = installer.is_installed().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_installer_preflight_check() {
    let installer = PytorchProfilerInstaller::new();
    let result = installer.preflight_check().await;
    assert!(result.is_ok());

    let checks = result.unwrap();
    assert!(!checks.is_empty());
}

#[test]
fn test_verification_result() {
    let installer = PytorchProfilerInstaller::new();
    let result = installer.verify_installation();
    assert!(result.is_ok());
}
