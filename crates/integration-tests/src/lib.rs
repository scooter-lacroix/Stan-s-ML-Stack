#[cfg(test)]
mod tests {
    use mlstack_hardware::HardwareDiscovery;
    use mlstack_installers::verification::VerificationModule;
    use mlstack_installers::{
        generate_ds_config, AiterInstaller, BitsAndBytesInstaller, DeepSpeedInstaller, Installer,
        MigraphxInstaller, PowerProfile, ProfilerConfig, PytorchProfilerInstaller,
        RocmSmiInstaller, WandbConfig, WandbInstaller, WandbMode, ZeroStage,
    };
    use std::process::Command;

    // ==========================================================================
    // Hardware Detection Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_hardware_discovery() {
        let discovery = HardwareDiscovery::new();
        let gpus = discovery.detect_gpus().expect("Hardware discovery failed");

        println!("Detected {} GPU(s)", gpus.len());
        for gpu in &gpus {
            println!("GPU: {} ({:?})", gpu.model, gpu.architecture);
        }
    }

    // ==========================================================================
    // PyTorch Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_pytorch_gpu() {
        // Ported from tests/pytorch_simple_test.py
        let output = Command::new("python3")
            .args(["-c", "import torch; print(torch.cuda.is_available())"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                println!("PyTorch CUDA available: {}", stdout.trim());

                if stdout.trim() == "True" {
                    // Try a simple operation
                    let op_output = Command::new("python3")
                        .args(["-c", "import torch; x = torch.ones(10, device='cuda'); print(x.sum().item())"])
                        .output()
                        .expect("Failed to run tensor op");
                    assert!(op_output.status.success());
                    assert_eq!(String::from_utf8_lossy(&op_output.stdout).trim(), "10.0");
                }
            }
            _ => {
                println!("PyTorch or Python3 not available, skipping functional check");
            }
        }
    }

    // ==========================================================================
    // Verification Module Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_verification_module() {
        let results = VerificationModule::run_all().await;
        assert!(!results.is_empty());
        for item in results {
            println!("{}: {:?}", item.name, item.status);
        }
    }

    // ==========================================================================
    // ONNX Runtime Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_onnx_rocm() {
        // Ported from tests/test_onnx_simple.py
        let output = Command::new("python3")
            .args(["-c", "import onnxruntime as ort; print('ROCMExecutionProvider' in ort.get_available_providers())"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                println!("ONNX ROCm available: {}", stdout.trim());
            }
            _ => {
                println!("ONNX Runtime or Python3 not available");
            }
        }
    }

    // ==========================================================================
    // AITER Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_aiter_installer_creation() {
        let installer = AiterInstaller::latest();
        assert_eq!(installer.name(), "AITER");
    }

    #[tokio::test]
    async fn test_aiter_preflight_check() {
        let installer = AiterInstaller::latest();
        let checks = installer.preflight_check().await;
        assert!(checks.is_ok());

        let checks = checks.unwrap();
        println!("AITER preflight checks:");
        for check in checks {
            println!("  - {}", check);
        }
    }

    #[tokio::test]
    async fn test_aiter_verification() {
        let installer = AiterInstaller::latest();
        let result = installer.verify_installation();

        if let Ok(verification) = result {
            println!("AITER verification:");
            println!("  Basic Import: {}", verification.basic_import);
            println!("  Version: {:?}", verification.version);
            println!("  GPU available: {}", verification.gpu_available);
        }
    }

    // ==========================================================================
    // DeepSpeed Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_deepspeed_installer_creation() {
        let installer = DeepSpeedInstaller::from_pypi();
        assert_eq!(installer.name(), "DeepSpeed");
    }

    #[tokio::test]
    async fn test_deepspeed_preflight_check() {
        let installer = DeepSpeedInstaller::from_pypi();
        let checks = installer.preflight_check().await;
        assert!(checks.is_ok());

        let checks = checks.unwrap();
        println!("DeepSpeed preflight checks:");
        for check in checks {
            println!("  - {}", check);
        }
    }

    #[tokio::test]
    async fn test_deepspeed_config_generation() {
        let config = generate_ds_config(ZeroStage::Stage2, 4, 16, true);

        assert!(config.get("zero_optimization").is_some());
        assert!(config.get("fp16").is_some());

        println!(
            "DeepSpeed config: {}",
            serde_json::to_string_pretty(&config).unwrap()
        );
    }

    #[tokio::test]
    async fn test_deepspeed_verification() {
        let installer = DeepSpeedInstaller::from_pypi();
        let result = installer.verify_installation();

        if let Ok(verification) = result {
            println!("DeepSpeed verification:");
            println!("  Installed: {}", verification.installed);
            println!("  Version: {:?}", verification.version);
            println!("  CUDA/HIP available: {}", verification.gpu_available);
        }
    }

    // ==========================================================================
    // BitsAndBytes Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_bitsandbytes_installer_creation() {
        let installer = BitsAndBytesInstaller::from_pypi();
        assert_eq!(installer.name(), "BitsAndBytes");
    }

    #[tokio::test]
    async fn test_bitsandbytes_preflight_check() {
        let installer = BitsAndBytesInstaller::from_pypi();
        let checks = installer.preflight_check().await;
        assert!(checks.is_ok());

        let checks = checks.unwrap();
        println!("BitsAndBytes preflight checks:");
        for check in checks {
            println!("  - {}", check);
        }
    }

    #[tokio::test]
    async fn test_bitsandbytes_verification() {
        let installer = BitsAndBytesInstaller::from_pypi();
        let result = installer.verify_installation();

        if let Ok(verification) = result {
            println!("BitsAndBytes verification:");
            println!("  Installed: {}", verification.installed);
            println!("  Version: {:?}", verification.version);
            println!("  INT8 supported: {}", verification.int8_supported);
            println!("  INT4 supported: {}", verification.int4_supported);
            println!("  GPU available: {}", verification.gpu_available);
        }
    }

    // ==========================================================================
    // ROCm SMI Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_rocm_smi_installer_creation() {
        let installer = RocmSmiInstaller::new();
        assert_eq!(installer.name(), "ROCm SMI");
    }

    #[tokio::test]
    async fn test_rocm_smi_preflight_check() {
        let installer = RocmSmiInstaller::new();
        let checks = installer.preflight_check().await;
        assert!(checks.is_ok());

        let checks = checks.unwrap();
        println!("ROCm SMI preflight checks:");
        for check in checks {
            println!("  - {}", check);
        }
    }

    #[tokio::test]
    async fn test_rocm_smi_list_gpus() {
        let installer = RocmSmiInstaller::new();
        let gpus = installer.list_gpus();

        if let Ok(gpu_list) = gpus {
            println!("Detected {} GPU(s) via rocm-smi:", gpu_list.len());
            for gpu in gpu_list {
                println!("  GPU {}: {} ({})", gpu.index, gpu.name, gpu.arch);
                if let Some(temp) = gpu.temperature {
                    println!("    Temperature: {}°C", temp);
                }
                if let Some(power) = gpu.power_usage {
                    println!("    Power: {}W", power);
                }
            }
        } else {
            println!("rocm-smi not available or no GPUs detected");
        }
    }

    #[tokio::test]
    async fn test_rocm_smi_power_profile() {
        let profile = PowerProfile::Compute;
        assert_eq!(profile.as_smi_arg(), "compute");
    }

    // ==========================================================================
    // MIGraphX Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_migraphx_installer_creation() {
        let installer = MigraphxInstaller::from_package_manager();
        assert_eq!(installer.name(), "MIGraphX");
    }

    #[tokio::test]
    async fn test_migraphx_preflight_check() {
        let installer = MigraphxInstaller::from_package_manager();
        let checks = installer.preflight_check().await;
        assert!(checks.is_ok());

        let checks = checks.unwrap();
        println!("MIGraphX preflight checks:");
        for check in checks {
            println!("  - {}", check);
        }
    }

    #[tokio::test]
    async fn test_migraphx_verification() {
        let installer = MigraphxInstaller::from_package_manager();
        let result = installer.verify_installation();

        if let Ok(verification) = result {
            println!("MIGraphX verification:");
            println!("  Installed: {}", verification.installed);
            println!("  Version: {:?}", verification.version);
            println!("  Python bindings: {}", verification.python_bindings);
            println!("  GPU available: {}", verification.gpu_available);
        }
    }

    // ==========================================================================
    // PyTorch Profiler Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_pytorch_profiler_installer_creation() {
        let installer = PytorchProfilerInstaller::new();
        assert_eq!(installer.name(), "PyTorch Profiler");
    }

    #[tokio::test]
    async fn test_pytorch_profiler_preflight_check() {
        let installer = PytorchProfilerInstaller::new();
        let checks = installer.preflight_check().await;
        assert!(checks.is_ok());

        let checks = checks.unwrap();
        println!("PyTorch Profiler preflight checks:");
        for check in checks {
            println!("  - {}", check);
        }
    }

    #[tokio::test]
    async fn test_pytorch_profiler_config_generation() {
        let config = ProfilerConfig::default();
        let code = config.generate_profiler_code();

        assert!(code.contains("profile"));
        assert!(code.contains("ProfilerActivity"));

        println!("Generated profiler code:\n{}", code);
    }

    #[tokio::test]
    async fn test_pytorch_profiler_verification() {
        let installer = PytorchProfilerInstaller::new();
        let result = installer.verify_installation();

        if let Ok(verification) = result {
            println!("PyTorch Profiler verification:");
            println!("  Profiler available: {}", verification.profiler_available);
            println!("  PyTorch version: {:?}", verification.pytorch_version);
            println!(
                "  TensorBoard available: {}",
                verification.tensorboard_available
            );
            println!("  GPU profiling: {}", verification.gpu_profiling);
        }
    }

    // ==========================================================================
    // Weights & Biases Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_wandb_installer_creation() {
        let installer = WandbInstaller::latest();
        assert_eq!(installer.name(), "Weights & Biases");
    }

    #[tokio::test]
    async fn test_wandb_preflight_check() {
        let installer = WandbInstaller::latest();
        let checks = installer.preflight_check().await;
        assert!(checks.is_ok());

        let checks = checks.unwrap();
        println!("W&B preflight checks:");
        for check in checks {
            println!("  - {}", check);
        }
    }

    #[tokio::test]
    async fn test_wandb_config_generation() {
        let mut config = WandbConfig::default();
        config.project = Some("test-project".to_string());
        config.entity = Some("test-entity".to_string());
        config.tags = vec!["test".to_string(), "integration".to_string()];

        let code = config.generate_init_code();

        assert!(code.contains("wandb.init"));
        assert!(code.contains("test-project"));
        assert!(code.contains("test-entity"));

        println!("Generated W&B init code:\n{}", code);
    }

    #[tokio::test]
    async fn test_wandb_env_map() {
        let mut config = WandbConfig::default();
        config.project = Some("test-project".to_string());
        config.mode = WandbMode::Offline;

        let env = config.as_env_map();

        assert_eq!(env.get("WANDB_MODE"), Some(&"offline".to_string()));
        assert_eq!(env.get("WANDB_PROJECT"), Some(&"test-project".to_string()));

        println!("W&B environment variables:");
        for (key, value) in &env {
            println!("  {}={}", key, value);
        }
    }

    #[tokio::test]
    async fn test_wandb_verification() {
        let installer = WandbInstaller::latest();
        let result = installer.verify_installation();

        if let Ok(verification) = result {
            println!("W&B verification:");
            println!("  Installed: {}", verification.installed);
            println!("  Version: {:?}", verification.version);
            println!("  CLI available: {}", verification.cli_available);
            println!("  Logged in: {}", verification.logged_in);
            println!("  GPU logging: {}", verification.gpu_logging);
        }
    }

    // ==========================================================================
    // Cross-Component Integration Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_all_installers_implement_trait() {
        // Verify all installers properly implement the Installer trait
        let installers: Vec<Box<dyn Installer + Send + Sync>> = vec![
            Box::new(AiterInstaller::latest()),
            Box::new(DeepSpeedInstaller::from_pypi()),
            Box::new(BitsAndBytesInstaller::from_pypi()),
            Box::new(RocmSmiInstaller::new()),
            Box::new(MigraphxInstaller::from_package_manager()),
            Box::new(PytorchProfilerInstaller::new()),
            Box::new(WandbInstaller::latest()),
        ];

        println!("Testing {} installers:", installers.len());
        for installer in &installers {
            println!("  - {} (version {})", installer.name(), installer.version());

            // Verify all methods are callable
            let is_installed = installer.is_installed().await;
            assert!(is_installed.is_ok());

            let preflight = installer.preflight_check().await;
            assert!(preflight.is_ok());
        }
    }

    #[tokio::test]
    async fn test_environment_configuration() {
        // Test that all environment configurations are valid
        use mlstack_installers::{
            AiterEnvironment, BitsAndBytesEnvironment, DeepSpeedEnvironment, MigraphxEnvironment,
            ProfilerEnvironment,
        };

        // let rocm_path = std::path::PathBuf::from("/opt/rocm");

        // AITER environment
        let aiter_env = AiterEnvironment::default();
        let aiter_map = aiter_env.as_env_map(&mlstack_installers::AiterBuildConfig::default());
        assert!(aiter_map.contains_key("ROCM_HOME"));

        // DeepSpeed environment
        let ds_env = DeepSpeedEnvironment::default();
        let ds_map = ds_env.as_env_map(&mlstack_installers::DeepSpeedBuildConfig::default());
        assert!(ds_map.contains_key("ROCM_PATH"));

        // BitsAndBytes environment
        let bnb_env = BitsAndBytesEnvironment::default();
        let bnb_map = bnb_env.as_env_map(&mlstack_installers::BitsAndBytesBuildConfig::default());
        assert!(bnb_map.contains_key("HSA_OVERRIDE_GFX_VERSION"));

        // MIGraphX environment
        let mgx_env = MigraphxEnvironment::default();
        let mgx_map = mgx_env.as_env_map();
        assert!(mgx_map.contains_key("ROCM_PATH"));

        // Profiler environment
        let prof_env = ProfilerEnvironment::default();
        let prof_map = prof_env.as_env_map();
        assert!(prof_map.contains_key("ROCM_PATH"));

        println!("All environment configurations validated successfully");
    }

    // ==========================================================================
    // Performance Benchmark Helper Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_benchmark_gpu_detection_time() {
        use std::time::Instant;

        let start = Instant::now();
        let installer = RocmSmiInstaller::new();
        let _ = installer.list_gpus();
        let duration = start.elapsed();

        println!("GPU detection time: {:?}", duration);
        assert!(duration.as_secs() < 10, "GPU detection took too long");
    }

    #[tokio::test]
    async fn test_benchmark_preflight_checks() {
        use std::time::Instant;

        let installers: Vec<(&str, Box<dyn Installer + Send + Sync>)> = vec![
            ("AITER", Box::new(AiterInstaller::from_pypi())),
            ("DeepSpeed", Box::new(DeepSpeedInstaller::from_pypi())),
            ("BitsAndBytes", Box::new(BitsAndBytesInstaller::from_pypi())),
            ("ROCm SMI", Box::new(RocmSmiInstaller::new())),
            (
                "MIGraphX",
                Box::new(MigraphxInstaller::from_package_manager()),
            ),
            (
                "PyTorch Profiler",
                Box::new(PytorchProfilerInstaller::new()),
            ),
            ("W&B", Box::new(WandbInstaller::latest())),
        ];

        println!("Preflight check benchmarks:");
        for (name, installer) in installers {
            let start = Instant::now();
            let _ = installer.preflight_check().await;
            let duration = start.elapsed();
            println!("  {}: {:?}", name, duration);
        }
    }
}
