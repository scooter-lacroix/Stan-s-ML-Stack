//! Anonymous telemetry payload construction.
//!
//! Builds a `TelemetryPayload` from benchmark results and hardware data.
//! The payload is strictly hardware data only — no personal information.
//!
//! # Privacy (VAL-TELE-005, VAL-TELE-006)
//!
//! - No username, hostname, home directory paths, email, IP addresses
//! - All payload keys are from a predefined allow-list of hardware/benchmark fields
//! - PII validation runs before any payload is finalized
//!
//! # Schema (VAL-TELE-007)
//!
//! - Payload conforms to JSON schema
//! - Serialized payload does not exceed 64 KiB

use crate::core::telemetry_types::{scan_json_for_personal_data, TelemetryPayload};
use crate::telemetry::benchmark::{BenchmarkOutcome, MAX_PAYLOAD_SIZE_BYTES};
use anyhow::{Context, Result};

/// Allow-list of top-level keys permitted in the telemetry payload.
/// No other keys are allowed — this prevents accidental inclusion of PII.
pub const PAYLOAD_ALLOW_LIST: &[&str] = &[
    "schema_version",
    "gpu_architecture",
    "gpu_model",
    "rocm_version",
    "gpu_count",
    "os_family",
    "cpu_cores",
    "ram_gb",
    "gpu_metrics",
    "benchmark",
];

/// Allow-list of keys permitted in the gpu_metrics sub-object.
const GPU_METRICS_ALLOW_LIST: &[&str] = &[
    "gpu_utilization_pct",
    "vram_used_mb",
    "vram_total_mb",
    "gpu_temp_c",
    "gpu_throttling",
    "gpu_clock_mhz",
];

/// Allow-list of keys permitted in the benchmark sub-object.
const BENCHMARK_ALLOW_LIST: &[&str] = &["duration_seconds", "completed", "sample_count"];

/// Build a telemetry payload from hardware info and benchmark outcome.
///
/// The payload contains only hardware/benchmark data from the allow-lists.
/// Personal data validation is performed before returning.
pub fn build_payload(
    gpu_architecture: impl Into<String>,
    gpu_model: impl Into<String>,
    rocm_version: impl Into<String>,
    gpu_count: usize,
    cpu_cores: usize,
    ram_gb: f32,
    benchmark_outcome: Option<&BenchmarkOutcome>,
) -> Result<TelemetryPayload> {
    let mut payload = TelemetryPayload::new(
        gpu_architecture,
        gpu_model,
        rocm_version,
        gpu_count,
        cpu_cores,
        ram_gb,
    );

    if let Some(outcome) = benchmark_outcome {
        payload.gpu_metrics = outcome.aggregated_metrics.clone();
        payload.benchmark = Some(outcome.benchmark_result.clone());
    }

    // Validate no personal data
    payload
        .validate_no_personal_data()
        .map_err(|errors| anyhow::anyhow!("Personal data detected in payload: {:?}", errors))?;

    Ok(payload)
}

/// Validate that a serialized payload conforms to all constraints:
///
/// 1. All top-level keys are from the allow-list (VAL-TELE-006)
/// 2. No personal data patterns in serialized JSON (VAL-TELE-005)
/// 3. Payload size ≤ 64 KiB (VAL-TELE-007)
/// 4. Valid JSON structure (VAL-TELE-007)
pub fn validate_payload(payload: &TelemetryPayload) -> Result<Vec<String>> {
    let mut warnings = Vec::new();

    // Serialize to JSON
    let json = serde_json::to_string(payload).context("Failed to serialize payload")?;
    let parsed: serde_json::Value =
        serde_json::from_str(&json).context("Payload is not valid JSON")?;

    // Check top-level keys against allow-list
    if let Some(obj) = parsed.as_object() {
        for key in obj.keys() {
            if !PAYLOAD_ALLOW_LIST.contains(&key.as_str()) {
                warnings.push(format!(
                    "Unexpected key '{}' in payload (not in allow-list)",
                    key
                ));
            }
        }
    }

    // Check gpu_metrics keys
    if let Some(metrics) = parsed.get("gpu_metrics").and_then(|m| m.as_object()) {
        for key in metrics.keys() {
            if !GPU_METRICS_ALLOW_LIST.contains(&key.as_str()) {
                warnings.push(format!(
                    "Unexpected key 'gpu_metrics.{}' (not in allow-list)",
                    key
                ));
            }
        }
    }

    // Check benchmark keys
    if let Some(bench) = parsed.get("benchmark").and_then(|b| b.as_object()) {
        for key in bench.keys() {
            if !BENCHMARK_ALLOW_LIST.contains(&key.as_str()) {
                warnings.push(format!(
                    "Unexpected key 'benchmark.{}' (not in allow-list)",
                    key
                ));
            }
        }
    }

    // Check for personal data patterns
    let pii_errors = scan_json_for_personal_data(&json);
    for err in pii_errors {
        warnings.push(format!("PII detected: {}", err));
    }

    // Check size constraint
    let size = json.len();
    if size > MAX_PAYLOAD_SIZE_BYTES {
        warnings.push(format!(
            "Payload size {} bytes exceeds maximum {} bytes",
            size, MAX_PAYLOAD_SIZE_BYTES
        ));
    }

    Ok(warnings)
}

/// Serialize a payload to JSON, ensuring it meets all constraints.
///
/// Returns the JSON string if valid, or an error describing the issues.
pub fn serialize_payload(payload: &TelemetryPayload) -> Result<String> {
    // Validate personal data
    payload
        .validate_no_personal_data()
        .map_err(|errors| anyhow::anyhow!("Personal data in payload: {:?}", errors))?;

    let json = serde_json::to_string(payload).context("Failed to serialize payload")?;

    // Validate allow-list
    let warnings = validate_payload(payload)?;
    if !warnings.is_empty() {
        anyhow::bail!("Payload validation warnings: {:?}", warnings);
    }

    // Validate size
    if json.len() > MAX_PAYLOAD_SIZE_BYTES {
        anyhow::bail!(
            "Payload size {} exceeds maximum {} bytes",
            json.len(),
            MAX_PAYLOAD_SIZE_BYTES
        );
    }

    Ok(json)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::telemetry_types::{BenchmarkResult, GpuMetrics};
    use crate::telemetry::benchmark::{BenchmarkOutcome, GpuSample};

    // =======================================================================
    // VAL-TELE-005: Anonymization — No Personal Data in Payload
    // =======================================================================

    #[test]
    fn test_payload_no_personal_data() {
        let payload = build_payload(
            "gfx1100",
            "AMD Radeon RX 7900 XTX",
            "7.2.1",
            1,
            16,
            54.0,
            None,
        )
        .expect("Should build clean payload");

        let json = serde_json::to_string(&payload).unwrap();
        let pii_errors = scan_json_for_personal_data(&json);
        assert!(
            pii_errors.is_empty(),
            "Payload should contain no PII, found: {:?}",
            pii_errors
        );
    }

    #[test]
    fn test_payload_rejects_home_path_in_gpu_model() {
        let result = build_payload(
            "gfx1100",
            "/home/user/gpu-model",
            "7.2.1",
            1,
            16,
            54.0,
            None,
        );
        assert!(result.is_err(), "Should reject home path in gpu_model");
    }

    #[test]
    fn test_payload_rejects_ip_in_gpu_arch() {
        let result = build_payload("192.168.1.1", "AMD GPU", "7.2.1", 1, 16, 54.0, None);
        assert!(result.is_err(), "Should reject IP address");
    }

    // =======================================================================
    // VAL-TELE-006: Anonymization — Hardware Data Only (Allow-list)
    // =======================================================================

    #[test]
    fn test_all_payload_keys_in_allow_list() {
        let payload = build_payload("gfx1100", "AMD GPU", "7.2.1", 1, 16, 54.0, None).unwrap();

        let warnings = validate_payload(&payload).unwrap();
        assert!(
            warnings.is_empty(),
            "All keys should be in allow-list, warnings: {:?}",
            warnings
        );
    }

    #[test]
    fn test_payload_with_benchmark_keys_in_allow_list() {
        let outcome = create_test_outcome(180, true, 19);
        let payload =
            build_payload("gfx1100", "AMD GPU", "7.2.1", 1, 16, 54.0, Some(&outcome)).unwrap();

        let warnings = validate_payload(&payload).unwrap();
        assert!(
            warnings.is_empty(),
            "All keys including benchmark should be in allow-list, warnings: {:?}",
            warnings
        );
    }

    #[test]
    fn test_gpu_metrics_keys_in_allow_list() {
        let mut payload = build_payload("gfx1100", "AMD GPU", "7.2.1", 1, 16, 54.0, None).unwrap();

        payload.gpu_metrics = Some(GpuMetrics {
            gpu_utilization_pct: 95.0,
            vram_used_mb: 8192.0,
            vram_total_mb: 24576.0,
            gpu_temp_c: 72.0,
            gpu_throttling: false,
            gpu_clock_mhz: 2500.0,
        });

        let warnings = validate_payload(&payload).unwrap();
        assert!(
            warnings.is_empty(),
            "All gpu_metrics keys should be in allow-list, warnings: {:?}",
            warnings
        );
    }

    // =======================================================================
    // VAL-TELE-007: Payload Schema — Structured and Bounded
    // =======================================================================

    #[test]
    fn test_payload_is_valid_json() {
        let payload = build_payload("gfx1100", "AMD GPU", "7.2.1", 1, 16, 54.0, None).unwrap();

        let json = serialize_payload(&payload).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object(), "Payload should be a JSON object");
    }

    #[test]
    fn test_payload_size_within_64kib() {
        let payload = build_payload(
            "gfx1100",
            "AMD Radeon RX 7900 XTX",
            "7.2.1",
            1,
            16,
            54.0,
            None,
        )
        .unwrap();

        let json = serialize_payload(&payload).unwrap();
        assert!(
            json.len() <= MAX_PAYLOAD_SIZE_BYTES,
            "Payload size {} should be ≤ {} bytes",
            json.len(),
            MAX_PAYLOAD_SIZE_BYTES
        );
        // A typical payload should be well under 1 KiB
        assert!(
            json.len() < 1024,
            "Typical payload should be under 1 KiB, got {} bytes",
            json.len()
        );
    }

    #[test]
    fn test_payload_size_with_benchmark_within_64kib() {
        let outcome = create_test_outcome(180, true, 19);
        let payload = build_payload(
            "gfx1100",
            "AMD Radeon RX 7900 XTX",
            "7.2.1",
            1,
            16,
            54.0,
            Some(&outcome),
        )
        .unwrap();

        let json = serialize_payload(&payload).unwrap();
        assert!(
            json.len() <= MAX_PAYLOAD_SIZE_BYTES,
            "Payload with benchmark should be ≤ {} bytes, got {}",
            MAX_PAYLOAD_SIZE_BYTES,
            json.len()
        );
    }

    #[test]
    fn test_payload_schema_has_required_fields() {
        let payload = build_payload("gfx1100", "AMD GPU", "7.2.1", 1, 16, 54.0, None).unwrap();

        let json = serde_json::to_string(&payload).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = parsed.as_object().unwrap();

        // Required fields
        assert!(obj.contains_key("schema_version"), "Missing schema_version");
        assert!(
            obj.contains_key("gpu_architecture"),
            "Missing gpu_architecture"
        );
        assert!(obj.contains_key("gpu_model"), "Missing gpu_model");
        assert!(obj.contains_key("rocm_version"), "Missing rocm_version");
        assert!(obj.contains_key("gpu_count"), "Missing gpu_count");
        assert!(obj.contains_key("os_family"), "Missing os_family");
        assert!(obj.contains_key("cpu_cores"), "Missing cpu_cores");
        assert!(obj.contains_key("ram_gb"), "Missing ram_gb");
    }

    #[test]
    fn test_benchmark_incomplete_flagged_in_payload() {
        let outcome = create_test_outcome(90, false, 10);
        let payload =
            build_payload("gfx1100", "AMD GPU", "7.2.1", 1, 16, 54.0, Some(&outcome)).unwrap();

        assert!(
            payload.benchmark.is_some(),
            "Benchmark should be present in payload"
        );
        let bench = payload.benchmark.unwrap();
        assert!(
            !bench.completed,
            "Incomplete benchmark should have completed=false"
        );
        assert_eq!(bench.duration_seconds, 90);
        assert_eq!(bench.sample_count, 10);
    }

    // =======================================================================
    // Full flow test
    // =======================================================================

    #[test]
    fn test_full_payload_build_and_validate() {
        let outcome = create_test_outcome(180, true, 19);
        let payload = build_payload(
            "gfx1100",
            "AMD Radeon RX 7900 XTX",
            "7.2.1",
            1,
            16,
            54.0,
            Some(&outcome),
        )
        .expect("Should build payload with benchmark");

        let json = serialize_payload(&payload).expect("Should serialize valid payload");

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object());

        // Verify no PII
        let pii = scan_json_for_personal_data(&json);
        assert!(pii.is_empty(), "No PII expected: {:?}", pii);

        // Verify allow-list compliance
        let warnings = validate_payload(&payload).unwrap();
        assert!(warnings.is_empty(), "No warnings expected: {:?}", warnings);

        // Verify size
        assert!(json.len() <= MAX_PAYLOAD_SIZE_BYTES);
    }

    // =======================================================================
    // Helpers
    // =======================================================================

    fn create_test_outcome(duration: u64, complete: bool, sample_count: usize) -> BenchmarkOutcome {
        let samples: Vec<GpuSample> = (0..sample_count)
            .map(|i| GpuSample {
                utilization_pct: 50.0,
                vram_used_mb: 8192.0,
                vram_total_mb: 24576.0,
                temp_c: 65.0,
                throttling: false,
                clock_mhz: 2000.0,
                timestamp_secs: i as f64 * 10.0,
            })
            .collect();

        let aggregated = if samples.is_empty() {
            None
        } else {
            Some(GpuMetrics {
                gpu_utilization_pct: 50.0,
                vram_used_mb: 8192.0,
                vram_total_mb: 24576.0,
                gpu_temp_c: 65.0,
                gpu_throttling: false,
                gpu_clock_mhz: 2000.0,
            })
        };

        let benchmark_result = BenchmarkResult {
            duration_seconds: duration,
            completed: complete,
            sample_count: sample_count as u32,
        };

        BenchmarkOutcome {
            samples,
            duration_secs: duration,
            is_complete: complete,
            aggregated_metrics: aggregated,
            benchmark_result,
        }
    }
}
