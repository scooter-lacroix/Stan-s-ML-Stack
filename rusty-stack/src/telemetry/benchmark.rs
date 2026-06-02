//! Stability benchmark for GPU hardware metrics.
//!
//! Runs a minimum 180-second benchmark collecting:
//! - GPU utilization (sampled every 10s, float in [0.0, 100.0])
//! - VRAM usage (used and total in MB)
//! - Thermal behavior (temperature in °C, range 0–120)
//! - Throttling indicators (boolean)
//! - Clock speeds (positive MHz)
//!
//! # Minimum Duration (VAL-TELE-001)
//!
//! The benchmark must run for at least 180 seconds. Shorter runs are flagged
//! as incomplete and excluded from submission.
//!
//! # Sampling (VAL-TELE-002)
//!
//! GPU utilization is sampled at least every 10 seconds. For a 180-second run,
//! this produces ≥18 samples.
//!
//! # Hardware Metrics (VAL-TELE-003, VAL-TELE-004)
//!
//! VRAM metrics enforce `vram_used_mb ≤ vram_total_mb`. Thermal values are
//! clamped to [0, 120]. Clock speeds are positive. Throttling is a boolean.

use crate::core::telemetry_types::{BenchmarkResult, GpuMetrics};
use anyhow::{Context, Result};
use std::process::Command;
use std::time::{Duration, Instant};

/// Minimum benchmark duration in seconds.
pub const MIN_BENCHMARK_DURATION_SECS: u64 = 180;

/// Sampling interval in seconds.
pub const SAMPLE_INTERVAL_SECS: u64 = 10;

/// Minimum number of samples required for a valid 180s benchmark.
/// At 10s intervals over 180s, we expect ≥18 samples (one at t=0 + 17 more).
pub const MIN_SAMPLE_COUNT: u32 = 18;

/// Maximum thermal value in Celsius.
pub const MAX_TEMP_C: f32 = 120.0;

/// Maximum payload size in bytes (64 KiB).
pub const MAX_PAYLOAD_SIZE_BYTES: usize = 65536;

/// A single GPU metric sample collected during a benchmark run.
#[derive(Debug, Clone)]
pub struct GpuSample {
    /// GPU utilization percentage (0.0–100.0).
    pub utilization_pct: f32,
    /// VRAM used in megabytes.
    pub vram_used_mb: f64,
    /// Total VRAM in megabytes.
    pub vram_total_mb: f64,
    /// GPU temperature in Celsius.
    pub temp_c: f32,
    /// Whether GPU thermal throttling is active.
    pub throttling: bool,
    /// GPU clock speed in MHz.
    pub clock_mhz: f32,
    /// Timestamp relative to benchmark start in seconds.
    pub timestamp_secs: f64,
}

/// Result of a benchmark run, including whether it meets minimum requirements.
#[derive(Debug, Clone)]
pub struct BenchmarkOutcome {
    /// All collected samples.
    pub samples: Vec<GpuSample>,
    /// Actual duration of the benchmark in seconds.
    pub duration_secs: u64,
    /// Whether the benchmark meets the minimum duration requirement.
    pub is_complete: bool,
    /// Aggregated GPU metrics (average of all samples).
    pub aggregated_metrics: Option<GpuMetrics>,
    /// Benchmark result for telemetry payload.
    pub benchmark_result: BenchmarkResult,
}

impl BenchmarkOutcome {
    /// Create a new benchmark outcome from collected samples.
    ///
    /// Validates that:
    /// - Duration ≥ 180 seconds
    /// - Sample count ≥ 18
    /// - All metrics are within valid ranges
    pub fn from_samples(samples: Vec<GpuSample>, duration_secs: u64) -> Self {
        let is_complete = duration_secs >= MIN_BENCHMARK_DURATION_SECS
            && samples.len() >= MIN_SAMPLE_COUNT as usize;

        let aggregated_metrics = if !samples.is_empty() {
            Some(aggregate_samples(&samples))
        } else {
            None
        };

        let benchmark_result = BenchmarkResult {
            duration_seconds: duration_secs,
            completed: is_complete,
            sample_count: samples.len() as u32,
        };

        Self {
            samples,
            duration_secs,
            is_complete,
            aggregated_metrics,
            benchmark_result,
        }
    }
}

/// Aggregate multiple GPU samples into a single GpuMetrics summary.
fn aggregate_samples(samples: &[GpuSample]) -> GpuMetrics {
    if samples.is_empty() {
        return GpuMetrics::default();
    }

    let count = samples.len() as f32;
    let count_f64 = samples.len() as f64;

    let avg_util = samples.iter().map(|s| s.utilization_pct).sum::<f32>() / count;
    let avg_vram_used = samples.iter().map(|s| s.vram_used_mb).sum::<f64>() / count_f64;
    let max_vram_total = samples
        .iter()
        .map(|s| s.vram_total_mb)
        .fold(0.0_f64, f64::max);
    let avg_temp = samples.iter().map(|s| s.temp_c).sum::<f32>() / count;
    let any_throttling = samples.iter().any(|s| s.throttling);
    let avg_clock = samples.iter().map(|s| s.clock_mhz).sum::<f32>() / count;

    GpuMetrics {
        gpu_utilization_pct: clamp_f32(avg_util, 0.0, 100.0),
        vram_used_mb: avg_vram_used,
        vram_total_mb: max_vram_total,
        gpu_temp_c: clamp_f32(avg_temp, 0.0, MAX_TEMP_C),
        gpu_throttling: any_throttling,
        gpu_clock_mhz: avg_clock.max(0.0),
    }
}

/// Clamp a float value to [min, max].
fn clamp_f32(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

/// Parse GPU utilization from rocm-smi output.
///
/// Expects output like: `GPU[0]\t\t: GPU use (%): 42`
fn parse_gpu_utilization(output: &str) -> f32 {
    for line in output.lines() {
        if line.contains("GPU use (%)") {
            if let Some(val) = line.split(':').next_back() {
                if let Ok(pct) = val.trim().parse::<f32>() {
                    return clamp_f32(pct, 0.0, 100.0);
                }
            }
        }
    }
    0.0
}

/// Parse VRAM usage from rocm-smi output.
///
/// Expects output like:
/// `GPU[0]\t\t: VRAM Total Memory (B): 25753026560`
/// `GPU[0]\t\t: VRAM Total Used Memory (B): 63635456`
fn parse_vram(output: &str) -> (f64, f64) {
    let mut total_bytes: f64 = 0.0;
    let mut used_bytes: f64 = 0.0;

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.contains("VRAM Total Memory (B)") {
            if let Some(val) = trimmed.split(':').next_back() {
                total_bytes = val.trim().parse::<f64>().unwrap_or(0.0);
            }
        } else if trimmed.contains("VRAM Total Used Memory (B)") {
            if let Some(val) = trimmed.split(':').next_back() {
                used_bytes = val.trim().parse::<f64>().unwrap_or(0.0);
            }
        }
    }

    let total_mb = total_bytes / (1024.0 * 1024.0);
    let used_mb = used_bytes / (1024.0 * 1024.0);
    (used_mb, total_mb)
}

/// Parse GPU temperature from rocm-smi output.
///
/// Expects output like: `GPU[0]\t\t: Temperature (Sensor edge) (C): 33.0`
fn parse_temperature(output: &str) -> f32 {
    for line in output.lines() {
        if line.contains("Temperature") && line.contains("(C)") {
            if let Some(val) = line.split(':').next_back() {
                if let Ok(temp) = val.trim().parse::<f32>() {
                    return clamp_f32(temp, 0.0, MAX_TEMP_C);
                }
            }
        }
    }
    0.0
}

/// Parse clock speed from rocm-smi output.
///
/// Expects output like: `GPU[0]\t\t: sclk clock level: S: (2500Mhz)`
fn parse_clock_mhz(output: &str) -> f32 {
    for line in output.lines() {
        if line.contains("sclk clock level") {
            // Extract value from parentheses like (2500Mhz)
            if let Some(start) = line.find('(') {
                if let Some(end) = line.find("Mhz)") {
                    let num_str = &line[start + 1..end];
                    if let Ok(mhz) = num_str.trim().parse::<f32>() {
                        return mhz.max(0.0);
                    }
                }
            }
        }
    }
    0.0
}

/// Detect throttling from rocm-smi output.
///
/// Throttling is inferred from performance level or explicit throttling indicators.
fn parse_throttling(output: &str) -> bool {
    let lower = output.to_lowercase();
    lower.contains("throttl") || lower.contains("performance level: manual")
}

/// Sample GPU metrics once using rocm-smi.
///
/// Queries rocm-smi for utilization, VRAM, temperature, clock, and throttling.
/// Returns a GpuSample with the collected data.
pub fn sample_gpu_metrics() -> Result<GpuSample> {
    let rocm_smi = find_rocm_smi()?;

    // Query all metrics in one call for efficiency
    let output = Command::new(&rocm_smi)
        .arg("--showuse")
        .arg("--showmeminfo")
        .arg("vram")
        .arg("--showtemp")
        .arg("--showclock")
        .output()
        .context("failed to execute rocm-smi")?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    let utilization_pct = parse_gpu_utilization(&stdout);
    let (vram_used_mb, vram_total_mb) = parse_vram(&stdout);
    let temp_c = parse_temperature(&stdout);
    let clock_mhz = parse_clock_mhz(&stdout);
    let throttling = parse_throttling(&stdout);

    Ok(GpuSample {
        utilization_pct,
        vram_used_mb,
        vram_total_mb,
        temp_c,
        throttling,
        clock_mhz,
        timestamp_secs: 0.0,
    })
}

/// Find the rocm-smi binary path.
fn find_rocm_smi() -> Result<String> {
    // Check standard locations
    let candidates = ["/opt/rocm/bin/rocm-smi", "/usr/bin/rocm-smi", "rocm-smi"];

    for candidate in &candidates {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return Ok(candidate.to_string());
        }
    }

    anyhow::bail!("rocm-smi not found in standard locations")
}

/// Run the stability benchmark for the specified duration.
///
/// This is the main entry point for benchmark execution. It collects GPU
/// metrics every `SAMPLE_INTERVAL_SECS` seconds for the given duration.
///
/// For a valid benchmark, `duration` must be ≥ `MIN_BENCHMARK_DURATION_SECS` (180s).
/// Shorter durations will produce an incomplete result.
///
/// # Arguments
///
/// * `duration` - How long to run the benchmark (minimum 180s for validity)
///
/// # Returns
///
/// A `BenchmarkOutcome` containing all samples and whether the benchmark is complete.
pub fn run_benchmark(duration: Duration) -> Result<BenchmarkOutcome> {
    let start = Instant::now();
    let mut samples: Vec<GpuSample> = Vec::new();

    // Take initial sample at t=0
    match sample_gpu_metrics() {
        Ok(mut sample) => {
            sample.timestamp_secs = 0.0;
            samples.push(sample);
        }
        Err(e) => {
            // Log but don't fail — we might recover on subsequent samples
            eprintln!("Warning: initial GPU sample failed: {}", e);
        }
    }

    // Sample every SAMPLE_INTERVAL_SECS
    let interval = Duration::from_secs(SAMPLE_INTERVAL_SECS);
    let mut next_sample = start + interval;

    while start.elapsed() < duration {
        let now = Instant::now();
        if now >= next_sample {
            match sample_gpu_metrics() {
                Ok(mut sample) => {
                    sample.timestamp_secs = now.duration_since(start).as_secs_f64();
                    samples.push(sample);
                }
                Err(e) => {
                    eprintln!("Warning: GPU sample failed at {:?}: {}", now - start, e);
                }
            }
            next_sample = now + interval;
        }

        // Sleep briefly to avoid busy-waiting
        std::thread::sleep(Duration::from_millis(100));
    }

    let elapsed = start.elapsed().as_secs();
    Ok(BenchmarkOutcome::from_samples(samples, elapsed))
}

/// Validate that a benchmark outcome meets all quality criteria.
///
/// Checks:
/// - Duration ≥ 180 seconds
/// - Sample count ≥ 18
/// - All utilization values in [0.0, 100.0]
/// - All VRAM used ≤ VRAM total
/// - All temperatures in [0, 120]
/// - All clock speeds positive
pub fn validate_benchmark(outcome: &BenchmarkOutcome) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // VAL-TELE-001: Minimum duration
    if outcome.duration_secs < MIN_BENCHMARK_DURATION_SECS {
        errors.push(format!(
            "Benchmark duration {}s is below minimum {}s",
            outcome.duration_secs, MIN_BENCHMARK_DURATION_SECS
        ));
    }

    // VAL-TELE-002: Minimum sample count
    if (outcome.samples.len() as u32) < MIN_SAMPLE_COUNT {
        errors.push(format!(
            "Sample count {} is below minimum {} for a valid benchmark",
            outcome.samples.len(),
            MIN_SAMPLE_COUNT
        ));
    }

    // Validate individual samples
    for (i, sample) in outcome.samples.iter().enumerate() {
        // Utilization bounds
        if !(0.0..=100.0).contains(&sample.utilization_pct) {
            errors.push(format!(
                "Sample {}: utilization {} out of range [0.0, 100.0]",
                i, sample.utilization_pct
            ));
        }

        // VAL-TELE-003: VRAM invariant
        if sample.vram_used_mb > sample.vram_total_mb {
            errors.push(format!(
                "Sample {}: vram_used_mb ({}) > vram_total_mb ({})",
                i, sample.vram_used_mb, sample.vram_total_mb
            ));
        }

        // VAL-TELE-004: Temperature range
        if !(0.0..=MAX_TEMP_C).contains(&sample.temp_c) {
            errors.push(format!(
                "Sample {}: temperature {} out of range [0, {}]",
                i, sample.temp_c, MAX_TEMP_C
            ));
        }

        // VAL-TELE-004: Clock speed positive
        if sample.clock_mhz < 0.0 {
            errors.push(format!(
                "Sample {}: clock speed {} is negative",
                i, sample.clock_mhz
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // VAL-TELE-001: Stability Benchmark — Minimum Duration
    // =======================================================================

    #[test]
    fn test_benchmark_rejects_short_duration() {
        // Create a synthetic outcome with only 90 seconds
        let samples = create_synthetic_samples(10, 90.0);
        let outcome = BenchmarkOutcome::from_samples(samples, 90);
        assert!(
            !outcome.is_complete,
            "90-second benchmark should be incomplete"
        );
        assert!(
            validate_benchmark(&outcome).is_err(),
            "90-second benchmark should fail validation"
        );
    }

    #[test]
    fn test_benchmark_accepts_180_seconds() {
        // Create a synthetic outcome with exactly 180 seconds and 19 samples
        let samples = create_synthetic_samples(19, 180.0);
        let outcome = BenchmarkOutcome::from_samples(samples, 180);
        assert!(
            outcome.is_complete,
            "180-second benchmark with 19 samples should be complete"
        );
        assert!(
            validate_benchmark(&outcome).is_ok(),
            "180-second benchmark should pass validation"
        );
    }

    #[test]
    fn test_benchmark_rejects_insufficient_samples() {
        // 180 seconds but only 10 samples (below 18 minimum)
        let samples = create_synthetic_samples(10, 180.0);
        let outcome = BenchmarkOutcome::from_samples(samples, 180);
        assert!(
            !outcome.is_complete,
            "180s benchmark with only 10 samples should be incomplete"
        );
        assert!(
            validate_benchmark(&outcome).is_err(),
            "Insufficient samples should fail validation"
        );
    }

    // =======================================================================
    // VAL-TELE-002: GPU Utilization Metric
    // =======================================================================

    #[test]
    fn test_gpu_utilization_in_valid_range() {
        let samples = create_synthetic_samples(19, 180.0);
        for sample in &samples {
            assert!(
                (0.0..=100.0).contains(&sample.utilization_pct),
                "Utilization {} out of range",
                sample.utilization_pct
            );
        }
    }

    #[test]
    fn test_sample_count_minimum_18_for_180s() {
        // At 10s intervals over 180s, we get 19 samples (t=0, 10, 20, ..., 180)
        let samples = create_synthetic_samples(19, 180.0);
        assert!(
            samples.len() >= MIN_SAMPLE_COUNT as usize,
            "Expected at least {} samples, got {}",
            MIN_SAMPLE_COUNT,
            samples.len()
        );
    }

    // =======================================================================
    // VAL-TELE-003: VRAM Metric
    // =======================================================================

    #[test]
    fn test_vram_used_le_total() {
        let samples = create_synthetic_samples(19, 180.0);
        for (i, sample) in samples.iter().enumerate() {
            assert!(
                sample.vram_used_mb <= sample.vram_total_mb,
                "Sample {}: vram_used_mb ({}) > vram_total_mb ({})",
                i,
                sample.vram_used_mb,
                sample.vram_total_mb
            );
        }
    }

    #[test]
    fn test_vram_validation_catches_used_exceeds_total() {
        let mut sample = create_synthetic_sample(0.0);
        sample.vram_used_mb = 30000.0;
        sample.vram_total_mb = 24576.0;
        let samples = vec![sample; 19];
        let outcome = BenchmarkOutcome::from_samples(samples, 180);
        let validation = validate_benchmark(&outcome);
        assert!(validation.is_err());
        let errors = validation.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("vram_used_mb")),
            "Should catch vram_used_mb > vram_total_mb"
        );
    }

    // =======================================================================
    // VAL-TELE-004: Thermal and Throttling Metrics
    // =======================================================================

    #[test]
    fn test_temperature_in_valid_range() {
        let samples = create_synthetic_samples(19, 180.0);
        for (i, sample) in samples.iter().enumerate() {
            assert!(
                (0.0..=MAX_TEMP_C).contains(&sample.temp_c),
                "Sample {}: temperature {} out of range [0, {}]",
                i,
                sample.temp_c,
                MAX_TEMP_C
            );
        }
    }

    #[test]
    fn test_throttling_is_bool() {
        let sample = create_synthetic_sample(0.0);
        // Just verify it compiles as bool
        let _: bool = sample.throttling;
    }

    #[test]
    fn test_clock_speed_positive() {
        let samples = create_synthetic_samples(19, 180.0);
        for (i, sample) in samples.iter().enumerate() {
            assert!(
                sample.clock_mhz >= 0.0,
                "Sample {}: clock speed {} is negative",
                i,
                sample.clock_mhz
            );
        }
    }

    // =======================================================================
    // Aggregation tests
    // =======================================================================

    #[test]
    fn test_aggregate_samples_averages_correctly() {
        let mut samples = Vec::new();
        for i in 0..3 {
            let s = GpuSample {
                utilization_pct: 50.0 + i as f32 * 10.0, // 50, 60, 70
                vram_used_mb: 1000.0,
                vram_total_mb: 24000.0,
                temp_c: 60.0,
                throttling: i == 2,
                clock_mhz: 2000.0,
                timestamp_secs: i as f64 * 10.0,
            };
            samples.push(s);
        }

        let metrics = aggregate_samples(&samples);
        // Average of 50, 60, 70 = 60
        assert!(
            (metrics.gpu_utilization_pct - 60.0).abs() < 0.1,
            "Expected avg utilization ~60, got {}",
            metrics.gpu_utilization_pct
        );
        assert!(
            metrics.gpu_throttling,
            "Any throttling should set throttling=true"
        );
        assert!(
            (metrics.vram_used_mb - 1000.0).abs() < 0.1,
            "Expected avg vram_used ~1000, got {}",
            metrics.vram_used_mb
        );
        assert!(
            (metrics.vram_total_mb - 24000.0).abs() < 0.1,
            "Expected max vram_total ~24000, got {}",
            metrics.vram_total_mb
        );
    }

    #[test]
    fn test_aggregate_empty_samples() {
        let metrics = aggregate_samples(&[]);
        assert_eq!(metrics, GpuMetrics::default());
    }

    // =======================================================================
    // Parsing tests
    // =======================================================================

    #[test]
    fn test_parse_gpu_utilization_valid() {
        let output = "GPU[0]\t\t: GPU use (%): 42";
        assert!((parse_gpu_utilization(output) - 42.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_gpu_utilization_zero() {
        let output = "GPU[0]\t\t: GPU use (%): 0";
        assert!((parse_gpu_utilization(output) - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_gpu_utilization_empty() {
        assert!((parse_gpu_utilization("") - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_vram_valid() {
        let output = "\
GPU[0]\t\t: VRAM Total Memory (B): 25753026560
GPU[0]\t\t: VRAM Total Used Memory (B): 63635456";
        let (used, total) = parse_vram(output);
        assert!(
            total > 24000.0,
            "Total VRAM should be ~24574 MB, got {}",
            total
        );
        assert!(used < total, "Used VRAM should be less than total");
    }

    #[test]
    fn test_parse_temperature_valid() {
        let output = "GPU[0]\t\t: Temperature (Sensor edge) (C): 33.0";
        assert!((parse_temperature(output) - 33.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_temperature_out_of_range_clamped() {
        let output = "GPU[0]\t\t: Temperature (Sensor edge) (C): 150.0";
        let temp = parse_temperature(output);
        assert!(
            temp <= MAX_TEMP_C,
            "Temperature should be clamped to {}, got {}",
            MAX_TEMP_C,
            temp
        );
    }

    #[test]
    fn test_parse_clock_mhz_valid() {
        let output = "GPU[0]\t\t: sclk clock level: S: (2500Mhz)";
        assert!((parse_clock_mhz(output) - 2500.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_throttling_detected() {
        let output = "GPU[0]\t\t: Throttling: active";
        assert!(parse_throttling(output));
    }

    #[test]
    fn test_parse_throttling_not_detected() {
        let output = "GPU[0]\t\t: Performance Level: auto";
        assert!(!parse_throttling(output));
    }

    // =======================================================================
    // Hardware test (requires real AMD GPU)
    // =======================================================================

    #[test]
    #[ignore]
    fn test_sample_gpu_metrics_real_hardware() {
        let sample = sample_gpu_metrics().expect("Should sample GPU metrics on real hardware");
        assert!(
            (0.0..=100.0).contains(&sample.utilization_pct),
            "Utilization out of range: {}",
            sample.utilization_pct
        );
        assert!(
            sample.vram_total_mb > 0.0,
            "VRAM total should be positive: {}",
            sample.vram_total_mb
        );
        assert!(
            sample.vram_used_mb <= sample.vram_total_mb,
            "VRAM used ({}) exceeds total ({})",
            sample.vram_used_mb,
            sample.vram_total_mb
        );
        assert!(
            (0.0..=MAX_TEMP_C).contains(&sample.temp_c),
            "Temperature out of range: {}",
            sample.temp_c
        );
    }

    // =======================================================================
    // Helpers
    // =======================================================================

    /// Create synthetic samples for testing.
    fn create_synthetic_samples(count: usize, _total_duration: f64) -> Vec<GpuSample> {
        (0..count)
            .map(|i| GpuSample {
                utilization_pct: 50.0,
                vram_used_mb: 8192.0,
                vram_total_mb: 24576.0,
                temp_c: 65.0,
                throttling: false,
                clock_mhz: 2000.0,
                timestamp_secs: i as f64 * SAMPLE_INTERVAL_SECS as f64,
            })
            .collect()
    }

    /// Create a single synthetic sample.
    fn create_synthetic_sample(timestamp: f64) -> GpuSample {
        GpuSample {
            utilization_pct: 50.0,
            vram_used_mb: 8192.0,
            vram_total_mb: 24576.0,
            temp_c: 65.0,
            throttling: false,
            clock_mhz: 2000.0,
            timestamp_secs: timestamp,
        }
    }
}
