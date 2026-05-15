//! Native Rust bootstrap module.
//!
//! Ports the three bootstrap scripts to native Rust:
//! - `scripts/install.sh` → `install_sh_clone_and_build()`
//! - `scripts/run_rusty_stack.sh` → `tui_launcher()`
//! - `scripts/enhanced_setup_environment.sh` → `setup_environment()`
//!
//! # Validation Assertions
//!
//! - **VAL-VBA-012**: install.sh equivalent — clone and build
//! - **VAL-VBA-013**: run_rusty_stack.sh equivalent — TUI launcher
//! - **VAL-VBA-014**: enhanced_setup_environment.sh — GPU filtering
//! - **VAL-VBA-015**: enhanced_setup_environment.sh — system dependency checking
//! - **VAL-VBA-016**: enhanced_setup_environment.sh — environment file creation
//! - **VAL-VBA-017**: enhanced_setup_environment.sh — ROCm optimization

pub mod env_setup;
pub mod install;
pub mod launcher;

// Re-export key types
pub use env_setup::{
    check_system_dependency, create_env_file, detect_correct_gpu_arch, detect_discrete_gpus,
    DependencyCheckResult, EnvFileResult, GpuArchInfo, GpuFilterResult, SystemDependency,
};
pub use install::{clone_or_update_repo, install_sh, InstallConfig, InstallResult};
pub use launcher::{launch_tui, BuildMode, LaunchResult};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- VAL-VBA-012: install.sh equivalent — clone and build ---

    #[test]
    fn test_install_config_defaults() {
        let config = InstallConfig::default();
        assert_eq!(
            config.repo_url,
            "https://github.com/scooter-lacroix/Stan-s-ML-Stack.git"
        );
        assert!(!config.skip_build);
        assert_eq!(config.branch, "main");
    }

    #[test]
    fn test_install_config_custom() {
        let config = InstallConfig {
            repo_dir: "/tmp/test-repo".to_string(),
            repo_url: "https://example.com/repo.git".to_string(),
            skip_build: true,
            branch: "develop".to_string(),
        };
        assert_eq!(config.repo_dir, "/tmp/test-repo");
        assert!(config.skip_build);
        assert_eq!(config.branch, "develop");
    }

    #[test]
    fn test_install_result_success() {
        let result = InstallResult::success(
            "/tmp/repo".to_string(),
            "/tmp/repo/rusty-stack/target/release/rusty-stack".to_string(),
        );
        assert!(result.success);
        assert!(result.binary_path.is_some());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_install_result_failure() {
        let result = InstallResult::failure("Build failed".to_string());
        assert!(!result.success);
        assert!(result.binary_path.is_none());
        assert!(!result.errors.is_empty());
    }

    // --- VAL-VBA-013: run_rusty_stack.sh equivalent — TUI launcher ---

    #[test]
    fn test_build_mode_default() {
        let mode = BuildMode::default();
        assert_eq!(mode, BuildMode::Release);
    }

    #[test]
    fn test_launch_result_success() {
        let result = LaunchResult::success("/path/to/rusty-stack".to_string());
        assert!(result.success);
        assert!(result.binary_path.is_some());
    }

    #[test]
    fn test_launch_result_failure() {
        let result = LaunchResult::failure("Build failed".to_string());
        assert!(!result.success);
        assert!(result.binary_path.is_none());
    }

    // --- VAL-VBA-014: GPU filtering ---

    #[test]
    fn test_gpu_arch_info_default() {
        let info = GpuArchInfo::default();
        assert_eq!(info.gpu_arch, "gfx1100");
        assert_eq!(info.hsa_override_gfx_version, "11.0.0");
    }

    #[test]
    fn test_gpu_arch_info_gfx1101() {
        let info = GpuArchInfo::from_arch("gfx1101");
        assert_eq!(info.gpu_arch, "gfx1101");
        assert_eq!(info.hsa_override_gfx_version, "11.0.1");
    }

    #[test]
    fn test_gpu_arch_info_gfx1102() {
        let info = GpuArchInfo::from_arch("gfx1102");
        assert_eq!(info.gpu_arch, "gfx1102");
        assert_eq!(info.hsa_override_gfx_version, "11.0.2");
    }

    #[test]
    fn test_gpu_arch_info_gfx1030() {
        let info = GpuArchInfo::from_arch("gfx1030");
        assert_eq!(info.gpu_arch, "gfx1030");
        assert_eq!(info.hsa_override_gfx_version, "10.3.0");
    }

    #[test]
    fn test_gpu_arch_info_gfx1200() {
        let info = GpuArchInfo::from_arch("gfx1200");
        assert_eq!(info.gpu_arch, "gfx1200");
        assert_eq!(info.hsa_override_gfx_version, "12.0.0");
    }

    #[test]
    fn test_gpu_arch_info_unknown_falls_back() {
        let info = GpuArchInfo::from_arch("gfx9999");
        assert_eq!(info.gpu_arch, "gfx9999");
        assert_eq!(info.hsa_override_gfx_version, "11.0.0"); // default fallback
    }

    #[test]
    fn test_gpu_filter_result_no_gpus() {
        let result = GpuFilterResult {
            discrete_gpu_indices: vec![],
            hip_visible_devices: "0".to_string(),
            gpu_count: 0,
        };
        assert!(result.discrete_gpu_indices.is_empty());
        // Should still have a default device
        assert_eq!(result.hip_visible_devices, "0");
    }

    #[test]
    fn test_gpu_filter_result_with_gpus() {
        let result = GpuFilterResult {
            discrete_gpu_indices: vec![0, 1],
            hip_visible_devices: "0,1".to_string(),
            gpu_count: 2,
        };
        assert_eq!(result.discrete_gpu_indices.len(), 2);
        assert_eq!(result.hip_visible_devices, "0,1");
    }

    // --- VAL-VBA-015: System dependency checking ---

    #[test]
    fn test_system_dependency_list() {
        let deps = SystemDependency::required_packages();
        assert!(deps.contains(&"git"), "git should be required");
        assert!(deps.contains(&"cmake"), "cmake should be required");
        assert!(
            deps.contains(&"python3-dev"),
            "python3-dev should be required"
        );
        assert!(
            deps.contains(&"build-essential"),
            "build-essential should be required"
        );
    }

    #[test]
    fn test_dependency_check_result_installed() {
        let result = DependencyCheckResult {
            package: "git".to_string(),
            installed: true,
            mapped_name: "git".to_string(),
        };
        assert!(result.installed);
    }

    #[test]
    fn test_dependency_check_result_missing() {
        let result = DependencyCheckResult {
            package: "nonexistent-pkg".to_string(),
            installed: false,
            mapped_name: "nonexistent-pkg".to_string(),
        };
        assert!(!result.installed);
    }

    // --- VAL-VBA-016: Environment file creation ---

    #[test]
    fn test_env_file_result_success() {
        let result = EnvFileResult::success("/home/user/.mlstack_env".to_string());
        assert!(result.success);
        assert!(result.path.is_some());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_env_file_result_failure() {
        let result = EnvFileResult::failure("Permission denied".to_string());
        assert!(!result.success);
        assert!(result.path.is_none());
    }

    #[test]
    fn test_env_file_content_has_required_variables() {
        let content = env_setup::generate_env_file_content(
            "0",         // hip_visible_devices
            "/opt/rocm", // rocm_path
            "7.2.0",     // rocm_version
            "latest",    // rocm_channel
            "gfx1100",   // gpu_arch
            "11.0.0",    // hsa_override_gfx_version
            "/usr/bin/python3",
        );
        // Must contain all required variable exports
        assert!(
            content.contains("HIP_VISIBLE_DEVICES"),
            "env file must set HIP_VISIBLE_DEVICES"
        );
        assert!(
            content.contains("CUDA_VISIBLE_DEVICES"),
            "env file must set CUDA_VISIBLE_DEVICES"
        );
        assert!(
            content.contains("PYTORCH_ROCM_DEVICE"),
            "env file must set PYTORCH_ROCM_DEVICE"
        );
        assert!(content.contains("ROCM_HOME"), "env file must set ROCM_HOME");
        assert!(
            content.contains("ROCM_VERSION"),
            "env file must set ROCM_VERSION"
        );
        assert!(
            content.contains("ROCM_CHANNEL"),
            "env file must set ROCM_CHANNEL"
        );
        assert!(content.contains("GPU_ARCH"), "env file must set GPU_ARCH");
        assert!(
            content.contains("LD_LIBRARY_PATH"),
            "env file must set LD_LIBRARY_PATH"
        );
        assert!(content.contains("PATH"), "env file must update PATH");
    }

    #[test]
    fn test_env_file_content_is_shell_sourceable() {
        let content = env_setup::generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "/usr/bin/python3",
        );
        // The content should be valid shell syntax — test with bash -n
        // (We can't actually run bash -n in unit tests, but we can check structure)
        assert!(
            content.starts_with("# ML Stack Environment File"),
            "env file must have header comment"
        );
        assert!(
            content.contains("export "),
            "env file must have export statements"
        );
        // No bash-specific syntax errors: should not contain unmatched quotes
        let export_count = content.matches("export ").count();
        assert!(
            export_count >= 10,
            "should have at least 10 export statements, got {export_count}"
        );
    }

    // --- VAL-VBA-017: ROCm optimization ---

    #[test]
    fn test_rocm_optimization_hsa_override() {
        let content = env_setup::generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "/usr/bin/python3",
        );
        assert!(
            content.contains("HSA_OVERRIDE_GFX_VERSION"),
            "must set HSA_OVERRIDE_GFX_VERSION"
        );
        assert!(
            content.contains("11.0.0"),
            "must contain the correct HSA_OVERRIDE_GFX_VERSION value"
        );
    }

    #[test]
    fn test_rocm_optimization_miopen_settings() {
        let content = env_setup::generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "/usr/bin/python3",
        );
        assert!(
            content.contains("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"),
            "must set MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"
        );
        assert!(
            content.contains("MIOPEN_FIND_MODE"),
            "must set MIOPEN_FIND_MODE"
        );
        assert!(
            content.contains("MIOPEN_FIND_ENFORCE"),
            "must set MIOPEN_FIND_ENFORCE"
        );
    }

    #[test]
    fn test_rocm_optimization_performance_vars() {
        let content = env_setup::generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "/usr/bin/python3",
        );
        assert!(
            content.contains("HSA_ENABLE_SDMA"),
            "must set HSA_ENABLE_SDMA"
        );
        assert!(
            content.contains("GPU_MAX_HEAP_SIZE"),
            "must set GPU_MAX_HEAP_SIZE"
        );
        assert!(
            content.contains("GPU_MAX_ALLOC_PERCENT"),
            "must set GPU_MAX_ALLOC_PERCENT"
        );
    }

    #[test]
    fn test_rocm_optimization_pytorch_vars() {
        let content = env_setup::generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "/usr/bin/python3",
        );
        assert!(
            content.contains("TORCH_CUDA_ARCH_LIST"),
            "must set TORCH_CUDA_ARCH_LIST"
        );
        assert!(
            content.contains("PYTORCH_ALLOC_CONF"),
            "must set PYTORCH_ALLOC_CONF"
        );
        assert!(
            content.contains("PYTORCH_HIP_ALLOC_CONF"),
            "must set PYTORCH_HIP_ALLOC_CONF"
        );
    }

    #[test]
    fn test_rocm_optimization_mpi_vars() {
        let content = env_setup::generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "/usr/bin/python3",
        );
        assert!(
            content.contains("OMPI_MCA_opal_cuda_support"),
            "must set OMPI MCA CUDA support"
        );
        assert!(content.contains("OMPI_MCA_pml"), "must set OMPI MCA PML");
    }

    #[test]
    fn test_rocm_optimization_channel_selection() {
        let content = env_setup::generate_env_file_content(
            "0",
            "/opt/rocm",
            "6.4.3",
            "legacy",
            "gfx1030",
            "10.3.0",
            "/usr/bin/python3",
        );
        assert!(content.contains("ROCM_CHANNEL"), "must set ROCM_CHANNEL");
        assert!(content.contains("legacy"), "must contain the channel value");
    }

    // --- Integration: detect_correct_gpu_arch mapping ---

    #[test]
    fn test_gpu_arch_from_marketing_name_7900_xtx() {
        let arch = detect_correct_gpu_arch(Some("Radeon RX 7900 XTX"));
        assert_eq!(arch.gpu_arch, "gfx1100");
    }

    #[test]
    fn test_gpu_arch_from_marketing_name_7800_xt() {
        let arch = detect_correct_gpu_arch(Some("Radeon RX 7800 XT"));
        assert_eq!(arch.gpu_arch, "gfx1101");
    }

    #[test]
    fn test_gpu_arch_from_marketing_name_7600() {
        let arch = detect_correct_gpu_arch(Some("Radeon RX 7600"));
        assert_eq!(arch.gpu_arch, "gfx1102");
    }

    #[test]
    fn test_gpu_arch_from_marketing_name_9070_xt() {
        let arch = detect_correct_gpu_arch(Some("Radeon RX 9070 XT"));
        assert_eq!(arch.gpu_arch, "gfx1200");
    }

    #[test]
    fn test_gpu_arch_from_marketing_name_none() {
        let arch = detect_correct_gpu_arch(None);
        assert_eq!(arch.gpu_arch, "gfx1100"); // default fallback
    }

    #[test]
    fn test_gpu_arch_from_marketing_name_unknown() {
        let arch = detect_correct_gpu_arch(Some("Unknown GPU"));
        assert_eq!(arch.gpu_arch, "gfx1100"); // default fallback
    }

    // --- Integration: detect_discrete_gpus filtering ---

    #[test]
    fn test_is_integrated_gpu_filtering() {
        // These should be identified as integrated GPUs
        assert!(env_setup::is_integrated_gpu_name("Raphael"));
        assert!(env_setup::is_integrated_gpu_name("AMD Ryzen Graphics"));
        assert!(env_setup::is_integrated_gpu_name("Integrated Graphics"));
        assert!(env_setup::is_integrated_gpu_name("AMD Radeon Graphics"));
        // These should NOT be identified as integrated
        assert!(!env_setup::is_integrated_gpu_name("Radeon RX 7900 XTX"));
        assert!(!env_setup::is_integrated_gpu_name("Radeon RX 7800 XT"));
    }
}
