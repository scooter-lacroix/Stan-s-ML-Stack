//! ML Stack Installer Benchmarks
//!
//! Criterion-based benchmarks for all installer components.
//! These benchmarks measure the performance of preflight checks,
//! verification, and configuration generation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mlstack_installers::{
    bitsandbytes::BitsAndBytesEnvironment,
    generate_ds_config,
    AiterBuildConfig,
    AiterEnvironment,
    // Core installers
    AiterInstaller,
    AiterSource,
    BitsAndBytesBuildConfig,
    BitsAndBytesInstaller,
    BitsAndBytesSource,
    DeepSpeedBuildConfig,
    DeepSpeedEnvironment,
    DeepSpeedInstaller,
    DeepSpeedSource,
    Installer,
    MigraphxBuildConfig,
    MigraphxEnvironment,
    MigraphxInstaller,
    MigraphxSource,
    PowerProfile,
    ProfilerConfig,
    ProfilerEnvironment,
    PytorchProfilerInstaller,
    RocmSmiInstaller,
    WandbConfig,
    WandbInstaller,
    WandbMode,
    ZeroStage,
};
use tokio::runtime::Runtime;

// =============================================================================
// Configuration Generation Benchmarks
// =============================================================================

fn bench_deepspeed_config_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_generation");

    for stage in [
        ZeroStage::Disabled,
        ZeroStage::Stage1,
        ZeroStage::Stage2,
        ZeroStage::Stage3,
    ] {
        group.bench_with_input(
            BenchmarkId::new("deepspeed_config", format!("{:?}", stage)),
            &stage,
            |b, stage| {
                b.iter(|| {
                    generate_ds_config(
                        black_box(*stage),
                        black_box(4),
                        black_box(16),
                        black_box(true),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_profiler_config_generation(c: &mut Criterion) {
    c.bench_function("profiler_config_generation", |b| {
        b.iter(|| {
            let config = ProfilerConfig::default();
            black_box(config.generate_profiler_code())
        });
    });
}

fn bench_wandb_config_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("wandb_config");

    group.bench_function("init_code", |b| {
        let mut config = WandbConfig::default();
        config.project = Some("test-project".to_string());
        config.entity = Some("test-entity".to_string());
        config.tags = vec!["test".to_string(), "benchmark".to_string()];

        b.iter(|| black_box(config.generate_init_code()));
    });

    group.bench_function("env_map", |b| {
        let mut config = WandbConfig::default();
        config.project = Some("test-project".to_string());
        config.mode = WandbMode::Offline;

        b.iter(|| black_box(config.as_env_map()));
    });

    group.finish();
}

// =============================================================================
// Environment Configuration Benchmarks
// =============================================================================

fn bench_environment_maps(c: &mut Criterion) {
    let mut group = c.benchmark_group("environment_maps");

    group.bench_function("aiter_env", |b| {
        let env = AiterEnvironment::default();
        let build = AiterBuildConfig::default();
        b.iter(|| black_box(env.as_env_map(&build)));
    });

    group.bench_function("deepspeed_env", |b| {
        let env = DeepSpeedEnvironment::default();
        let build = DeepSpeedBuildConfig::default();
        b.iter(|| black_box(env.as_env_map(&build)));
    });

    group.bench_function("bitsandbytes_env", |b| {
        let env = BitsAndBytesEnvironment::default();
        let build = BitsAndBytesBuildConfig::default();
        b.iter(|| black_box(env.as_env_map(&build)));
    });

    group.bench_function("migraphx_env", |b| {
        let env = MigraphxEnvironment::default();
        b.iter(|| black_box(env.as_env_map()));
    });

    group.bench_function("profiler_env", |b| {
        let env = ProfilerEnvironment::default();
        b.iter(|| black_box(env.as_env_map()));
    });

    group.finish();
}

// =============================================================================
// Installer Creation Benchmarks
// =============================================================================

fn bench_installer_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("installer_creation");

    group.bench_function("aiter", |b| {
        b.iter(|| black_box(AiterInstaller::from_pypi()));
    });

    group.bench_function("deepspeed", |b| {
        b.iter(|| black_box(DeepSpeedInstaller::from_pypi()));
    });

    group.bench_function("bitsandbytes", |b| {
        b.iter(|| black_box(BitsAndBytesInstaller::from_pypi()));
    });

    group.bench_function("rocm_smi", |b| {
        b.iter(|| black_box(RocmSmiInstaller::new()));
    });

    group.bench_function("migraphx", |b| {
        b.iter(|| black_box(MigraphxInstaller::from_package_manager()));
    });

    group.bench_function("pytorch_profiler", |b| {
        b.iter(|| black_box(PytorchProfilerInstaller::new()));
    });

    group.bench_function("wandb", |b| {
        b.iter(|| black_box(WandbInstaller::latest()));
    });

    group.finish();
}

// =============================================================================
// Async Preflight Check Benchmarks
// =============================================================================

fn bench_preflight_checks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("preflight_checks");

    // Increase sample size for more stable results
    group.sample_size(20);

    group.bench_function("aiter", |b| {
        let installer = AiterInstaller::from_pypi();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.preflight_check().await) });
    });

    group.bench_function("deepspeed", |b| {
        let installer = DeepSpeedInstaller::from_pypi();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.preflight_check().await) });
    });

    group.bench_function("bitsandbytes", |b| {
        let installer = BitsAndBytesInstaller::from_pypi();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.preflight_check().await) });
    });

    group.bench_function("rocm_smi", |b| {
        let installer = RocmSmiInstaller::new();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.preflight_check().await) });
    });

    group.bench_function("migraphx", |b| {
        let installer = MigraphxInstaller::from_package_manager();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.preflight_check().await) });
    });

    group.bench_function("pytorch_profiler", |b| {
        let installer = PytorchProfilerInstaller::new();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.preflight_check().await) });
    });

    group.bench_function("wandb", |b| {
        let installer = WandbInstaller::latest();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.preflight_check().await) });
    });

    group.finish();
}

// =============================================================================
// Is Installed Check Benchmarks
// =============================================================================

fn bench_is_installed_checks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("is_installed_checks");

    group.sample_size(20);

    group.bench_function("aiter", |b| {
        let installer = AiterInstaller::from_pypi();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.is_installed().await) });
    });

    group.bench_function("deepspeed", |b| {
        let installer = DeepSpeedInstaller::from_pypi();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.is_installed().await) });
    });

    group.bench_function("bitsandbytes", |b| {
        let installer = BitsAndBytesInstaller::from_pypi();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.is_installed().await) });
    });

    group.bench_function("rocm_smi", |b| {
        let installer = RocmSmiInstaller::new();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.is_installed().await) });
    });

    group.bench_function("migraphx", |b| {
        let installer = MigraphxInstaller::from_package_manager();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.is_installed().await) });
    });

    group.bench_function("pytorch_profiler", |b| {
        let installer = PytorchProfilerInstaller::new();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.is_installed().await) });
    });

    group.bench_function("wandb", |b| {
        let installer = WandbInstaller::latest();
        b.to_async(&rt)
            .iter(|| async { black_box(installer.is_installed().await) });
    });

    group.finish();
}

// =============================================================================
// GPU-Specific Benchmarks
// =============================================================================

fn bench_rocm_smi_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("rocm_smi_operations");

    group.sample_size(10);

    group.bench_function("list_gpus", |b| {
        let installer = RocmSmiInstaller::new();
        b.iter(|| black_box(installer.list_gpus()));
    });

    group.bench_function("power_profile_conversion", |b| {
        let profiles = [
            PowerProfile::Auto,
            PowerProfile::Low,
            PowerProfile::High,
            PowerProfile::Compute,
        ];
        let mut idx = 0;

        b.iter(|| {
            let profile = profiles[idx % profiles.len()];
            idx += 1;
            black_box(profile.as_smi_arg())
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark Groups
// =============================================================================

criterion_group!(
    config_benches,
    bench_deepspeed_config_generation,
    bench_profiler_config_generation,
    bench_wandb_config_generation,
);

criterion_group!(env_benches, bench_environment_maps,);

criterion_group!(creation_benches, bench_installer_creation,);

criterion_group!(
    async_benches,
    bench_preflight_checks,
    bench_is_installed_checks,
);

criterion_group!(gpu_benches, bench_rocm_smi_operations,);

criterion_main!(
    config_benches,
    env_benches,
    creation_benches,
    async_benches,
    gpu_benches,
);
