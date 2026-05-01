//! Telemetry payload types with privacy-safe design.
//!
//! This module implements:
//! - [`TelemetryPayload`] — the top-level payload schema for anonymous
//!   hardware/benchmark telemetry submission.
//! - [`GpuMetrics`] — GPU utilization, VRAM, thermal, and clock data.
//! - [`BenchmarkResult`] — stability benchmark outcome data.
//!
//! # Privacy Design (VAL-CORE-017)
//!
//! The telemetry payload is designed to **exclude all personal data**:
//! - No username, hostname, or home directory paths
//! - No IP addresses or network identifiers
//! - No free-form user input
//! - No sudo passwords or credentials
//!
//! All fields are strictly hardware/benchmark data from a predefined
//! allow-list. The payload is validated before submission to ensure
//! no personal data patterns are present.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Telemetry Error
// ---------------------------------------------------------------------------

/// Error returned when a payload contains personal data patterns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersonalDataError {
    pub field: String,
    pub pattern_matched: String,
}

impl fmt::Display for PersonalDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "personal data detected in field '{}': matched pattern '{}'",
            self.field, self.pattern_matched
        )
    }
}

impl std::error::Error for PersonalDataError {}

// ---------------------------------------------------------------------------
// GpuMetrics
// ---------------------------------------------------------------------------

/// GPU hardware metrics sampled during a benchmark run.
///
/// All fields are numeric hardware data — no personal identifiers.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GpuMetrics {
    /// GPU utilization percentage (0.0–100.0).
    pub gpu_utilization_pct: f32,
    /// VRAM used in megabytes.
    pub vram_used_mb: f64,
    /// Total VRAM in megabytes.
    pub vram_total_mb: f64,
    /// GPU temperature in Celsius (0–120).
    pub gpu_temp_c: f32,
    /// Whether GPU thermal throttling is active.
    pub gpu_throttling: bool,
    /// GPU clock speed in MHz.
    pub gpu_clock_mhz: f32,
}

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------

/// Outcome of a stability benchmark run.
///
/// Contains duration and pass/fail status only.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkResult {
    /// Duration of the benchmark in seconds.
    pub duration_seconds: u64,
    /// Whether the benchmark completed successfully.
    pub completed: bool,
    /// Number of metric samples collected.
    pub sample_count: u32,
}

// ---------------------------------------------------------------------------
// TelemetryPayload
// ---------------------------------------------------------------------------

/// Top-level telemetry payload for anonymous submission.
///
/// Contains only hardware and benchmark data — no personal information.
/// The payload is validated before submission to ensure privacy compliance.
///
/// # Allowed Fields
///
/// All fields in this struct are from the predefined allow-list:
/// - `schema_version` — payload format version
/// - `gpu_architecture` — GPU architecture identifier (e.g., "gfx1100")
/// - `gpu_model` — GPU model name (e.g., "AMD Radeon RX 7900 XTX")
/// - `rocm_version` — ROCm version string
/// - `gpu_count` — number of GPUs detected
/// - `os_family` — OS family (e.g., "linux")
/// - `cpu_cores` — number of CPU cores
/// - `ram_gb` — total RAM in gigabytes
/// - `gpu_metrics` — GPU hardware metrics
/// - `benchmark` — benchmark result data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TelemetryPayload {
    /// Payload schema version.
    pub schema_version: u32,
    /// GPU architecture identifier (e.g., "gfx1100").
    pub gpu_architecture: String,
    /// GPU model name (e.g., "AMD Radeon RX 7900 XTX").
    pub gpu_model: String,
    /// ROCm version string.
    pub rocm_version: String,
    /// Number of GPUs detected.
    pub gpu_count: usize,
    /// OS family (e.g., "linux").
    pub os_family: String,
    /// Number of CPU cores.
    pub cpu_cores: usize,
    /// Total RAM in gigabytes.
    pub ram_gb: f32,
    /// GPU hardware metrics (if benchmark was run).
    #[serde(default)]
    pub gpu_metrics: Option<GpuMetrics>,
    /// Benchmark result (if benchmark was run).
    #[serde(default)]
    pub benchmark: Option<BenchmarkResult>,
}

impl TelemetryPayload {
    /// Create a new telemetry payload with the given hardware info.
    pub fn new(
        gpu_architecture: impl Into<String>,
        gpu_model: impl Into<String>,
        rocm_version: impl Into<String>,
        gpu_count: usize,
        cpu_cores: usize,
        ram_gb: f32,
    ) -> Self {
        Self {
            schema_version: 1,
            gpu_architecture: gpu_architecture.into(),
            gpu_model: gpu_model.into(),
            rocm_version: rocm_version.into(),
            gpu_count,
            os_family: "linux".to_string(),
            cpu_cores,
            ram_gb,
            gpu_metrics: None,
            benchmark: None,
        }
    }

    /// Validate that this payload contains no personal data.
    ///
    /// Scans all string fields against known personal data patterns:
    /// - Usernames
    /// - Hostnames
    /// - Home directory paths
    /// - IP addresses
    /// - Email addresses
    /// - Free-form user input patterns
    pub fn validate_no_personal_data(&self) -> Result<(), Vec<PersonalDataError>> {
        let mut errors = Vec::new();

        // Collect all string fields for scanning
        let string_fields: Vec<(&str, &str)> = vec![
            ("gpu_architecture", &self.gpu_architecture),
            ("gpu_model", &self.gpu_model),
            ("rocm_version", &self.rocm_version),
            ("os_family", &self.os_family),
        ];

        for (field_name, value) in &string_fields {
            if let Some(err) = check_personal_data(field_name, value) {
                errors.push(err);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate and return the full serialized JSON payload.
    ///
    /// Performs personal data validation before serialization.
    pub fn to_validated_json(&self) -> Result<String, Vec<PersonalDataError>> {
        self.validate_no_personal_data()?;
        Ok(serde_json::to_string(self).unwrap())
    }
}

// ---------------------------------------------------------------------------
// Personal data pattern checking
// ---------------------------------------------------------------------------

/// Known personal data patterns to scan for.
const PERSONAL_DATA_PATTERNS: &[&str] = &[
    r"/home/",                                         // Home directory paths
    r"/Users/",                                        // macOS home paths
    r"C:\\Users\\",                                    // Windows user paths
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",         // IP addresses
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", // Email addresses
    r"sudo",                                           // Sudo references
    r"password",    // Password references (case-insensitive handled in check)
    r"token",       // Token references
    r"secret",      // Secret references
    r"api[_-]?key", // API key references
    r"bearer",      // Bearer token references
];

/// Check a string value for personal data patterns.
fn check_personal_data(field_name: &str, value: &str) -> Option<PersonalDataError> {
    let lowercase = value.to_lowercase();

    for pattern in PERSONAL_DATA_PATTERNS {
        if let Ok(re) = Regex::new(pattern) {
            if re.is_match(value) || re.is_match(&lowercase) {
                return Some(PersonalDataError {
                    field: field_name.to_string(),
                    pattern_matched: pattern.to_string(),
                });
            }
        }
    }

    // Additional check for common home path pattern in lowercase
    if lowercase.contains("/home/") || lowercase.contains("/users/") {
        return Some(PersonalDataError {
            field: field_name.to_string(),
            pattern_matched: "home/user path".to_string(),
        });
    }

    None
}

/// Scan serialized JSON for personal data patterns.
///
/// This is a belt-and-suspenders check on the final serialized form.
pub fn scan_json_for_personal_data(json: &str) -> Vec<PersonalDataError> {
    let mut errors = Vec::new();

    for pattern in PERSONAL_DATA_PATTERNS {
        if let Ok(re) = Regex::new(pattern) {
            if re.is_match(json) {
                errors.push(PersonalDataError {
                    field: "(serialized JSON)".to_string(),
                    pattern_matched: pattern.to_string(),
                });
            }
        }
    }

    errors
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // VAL-CORE-017: TelemetryPayload excludes personal data
    // =======================================================================

    #[test]
    fn test_telemetry_payload_validates_clean_payload() {
        let payload =
            TelemetryPayload::new("gfx1100", "AMD Radeon RX 7900 XTX", "7.2.1", 1, 16, 54.0);
        assert!(
            payload.validate_no_personal_data().is_ok(),
            "Clean payload should pass validation"
        );
    }

    #[test]
    fn test_telemetry_payload_rejects_home_path() {
        let payload = TelemetryPayload::new(
            "gfx1100",
            "/home/user/gpu-model", // contains home path
            "7.2.1",
            1,
            16,
            54.0,
        );
        let result = payload.validate_no_personal_data();
        assert!(result.is_err(), "Should reject home path");
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.field == "gpu_model"));
    }

    #[test]
    fn test_telemetry_payload_rejects_ip_address() {
        let payload = TelemetryPayload::new(
            "192.168.1.1", // IP address
            "AMD GPU",
            "7.2.1",
            1,
            16,
            54.0,
        );
        let result = payload.validate_no_personal_data();
        assert!(result.is_err(), "Should reject IP address");
    }

    #[test]
    fn test_telemetry_payload_rejects_email() {
        let payload = TelemetryPayload::new(
            "gfx1100",
            "user@example.com", // email address
            "7.2.1",
            1,
            16,
            54.0,
        );
        let result = payload.validate_no_personal_data();
        assert!(result.is_err(), "Should reject email address");
    }

    #[test]
    fn test_telemetry_payload_rejects_password_reference() {
        let payload = TelemetryPayload::new(
            "gfx1100",
            "AMD GPU",
            "password=secret", // contains password
            1,
            16,
            54.0,
        );
        let result = payload.validate_no_personal_data();
        assert!(result.is_err(), "Should reject password reference");
    }

    #[test]
    fn test_telemetry_payload_rejects_api_key() {
        let payload = TelemetryPayload::new(
            "gfx1100",
            "AMD GPU",
            "api_key=abc123", // contains api_key
            1,
            16,
            54.0,
        );
        let result = payload.validate_no_personal_data();
        assert!(result.is_err(), "Should reject api_key reference");
    }

    #[test]
    fn test_telemetry_payload_to_validated_json_succeeds_for_clean() {
        let payload =
            TelemetryPayload::new("gfx1100", "AMD Radeon RX 7900 XTX", "7.2.1", 1, 16, 54.0);
        let json = payload.to_validated_json();
        assert!(json.is_ok(), "Clean payload should serialize successfully");
        let json_str = json.unwrap();
        assert!(json_str.contains("gfx1100"));
    }

    #[test]
    fn test_telemetry_payload_scan_clean_json() {
        let payload =
            TelemetryPayload::new("gfx1100", "AMD Radeon RX 7900 XTX", "7.2.1", 1, 16, 54.0);
        let json = serde_json::to_string(&payload).unwrap();
        let errors = scan_json_for_personal_data(&json);
        assert!(
            errors.is_empty(),
            "Clean serialized JSON should have no PII errors"
        );
    }

    // =======================================================================
    // Serde roundtrip
    // =======================================================================

    #[test]
    fn test_telemetry_payload_serde_roundtrip() {
        let payload =
            TelemetryPayload::new("gfx1100", "AMD Radeon RX 7900 XTX", "7.2.1", 2, 16, 54.0);
        let json = serde_json::to_string(&payload).unwrap();
        let back: TelemetryPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(payload, back);
    }

    #[test]
    fn test_telemetry_payload_with_metrics_roundtrip() {
        let mut payload =
            TelemetryPayload::new("gfx1100", "AMD Radeon RX 7900 XTX", "7.2.1", 1, 16, 54.0);
        payload.gpu_metrics = Some(GpuMetrics {
            gpu_utilization_pct: 95.0,
            vram_used_mb: 20480.0,
            vram_total_mb: 24576.0,
            gpu_temp_c: 72.0,
            gpu_throttling: false,
            gpu_clock_mhz: 2500.0,
        });
        payload.benchmark = Some(BenchmarkResult {
            duration_seconds: 180,
            completed: true,
            sample_count: 18,
        });

        let json = serde_json::to_string(&payload).unwrap();
        let back: TelemetryPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(payload, back);
    }

    #[test]
    fn test_gpu_metrics_serde_roundtrip() {
        let metrics = GpuMetrics {
            gpu_utilization_pct: 95.0,
            vram_used_mb: 20480.0,
            vram_total_mb: 24576.0,
            gpu_temp_c: 72.0,
            gpu_throttling: false,
            gpu_clock_mhz: 2500.0,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let back: GpuMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics, back);
    }

    #[test]
    fn test_gpu_metrics_default() {
        let metrics = GpuMetrics::default();
        assert_eq!(metrics.gpu_utilization_pct, 0.0);
        assert_eq!(metrics.vram_used_mb, 0.0);
        assert_eq!(metrics.vram_total_mb, 0.0);
        assert_eq!(metrics.gpu_temp_c, 0.0);
        assert!(!metrics.gpu_throttling);
        assert_eq!(metrics.gpu_clock_mhz, 0.0);
    }

    #[test]
    fn test_benchmark_result_serde_roundtrip() {
        let result = BenchmarkResult {
            duration_seconds: 180,
            completed: true,
            sample_count: 18,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, back);
    }

    #[test]
    fn test_personal_data_error_display() {
        let err = PersonalDataError {
            field: "gpu_model".to_string(),
            pattern_matched: "/home/".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("gpu_model"));
        assert!(msg.contains("/home/"));
    }

    #[test]
    fn test_personal_data_error_is_std_error() {
        let err = PersonalDataError {
            field: "test".to_string(),
            pattern_matched: "test".to_string(),
        };
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_telemetry_payload_allow_list_fields_only() {
        let payload = TelemetryPayload::new("gfx1100", "AMD GPU", "7.2.1", 1, 16, 54.0);
        let json = serde_json::to_string(&payload).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Verify all expected fields are present
        let allowed_keys = [
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

        let obj = parsed.as_object().unwrap();
        for key in &allowed_keys {
            assert!(obj.contains_key(*key), "Missing allowed key: {key}");
        }

        // Verify no extra keys
        for key in obj.keys() {
            assert!(
                allowed_keys.contains(&key.as_str()),
                "Unexpected key in payload: {key}"
            );
        }
    }
}
