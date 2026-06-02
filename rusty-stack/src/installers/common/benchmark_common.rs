//! Benchmark common utilities — PCI bus ID normalization, timing, result
//! structs with JSON serialization, and timestamped logging.
//!
//! Ports the core infrastructure from `scripts/lib/benchmark_common.sh`:
//! - PCI bus ID normalization (`0000:01:00.0` → `01:00.0`)
//! - Benchmark timing utilities (elapsed calculation, formatting)
//! - `BenchmarkResult` struct with JSON serialization compatible with
//!   `benchmark_logs.rs` parsing
//! - Timestamped logging with `[YYYY-MM-DD HH:MM:SS]` prefix
//! - Tee-style logging to stderr + optional file output
//!
//! # Validation Assertions
//!
//! - **VAL-INFRA-013**: PCI bus normalization, timing utilities, result struct
//!   with JSON serialization
//! - **VAL-INFRA-014**: Timestamped logging `[YYYY-MM-DD HH:MM:SS]` prefix,
//!   supports stderr + file output

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ===========================================================================
// PCI Bus ID Normalization
// ===========================================================================

/// Normalize a PCI bus ID string.
///
/// Strips leading `0000:` domain prefix, lowercases the input, and validates
/// the result matches the pattern `XX:XX.X` (where X is a hex digit and the
/// function number is 0–7).
///
/// # Examples
///
/// ```
/// use rusty_stack::installers::common::benchmark_common::normalize_pci_bus_id;
///
/// assert_eq!(normalize_pci_bus_id("0000:01:00.0"), Ok("01:00.0".to_string()));
/// assert_eq!(normalize_pci_bus_id("01:00.0"), Ok("01:00.0".to_string()));
/// assert_eq!(normalize_pci_bus_id("03:00.0"), Ok("03:00.0".to_string()));
/// ```
///
/// # Validation
///
/// - **VAL-INFRA-013**: PCI bus normalization (0000:01:00.0 → 01:00.0)
pub fn normalize_pci_bus_id(raw: &str) -> Result<String, String> {
    let mut s = raw.trim().to_lowercase();
    // Strip leading whitespace and take only the first token
    s = s.split_whitespace().next().unwrap_or("").to_string();
    // Strip the 0000: domain prefix if present
    if s.starts_with("0000:") {
        s = s[5..].to_string();
    }
    // Validate: XX:XX.X where X is hex and function is 0-7
    if is_valid_pci_bus_id(&s) {
        Ok(s)
    } else {
        Err(format!("Invalid PCI bus ID: '{s}'"))
    }
}

/// Check whether a string matches the PCI bus ID format `XX:XX.X`.
fn is_valid_pci_bus_id(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() != 7 {
        return false;
    }
    // Format: hh:hh.f
    //         0123456
    is_hex_digit(bytes[0])
        && is_hex_digit(bytes[1])
        && bytes[2] == b':'
        && is_hex_digit(bytes[3])
        && is_hex_digit(bytes[4])
        && bytes[5] == b'.'
        && bytes[6] >= b'0'
        && bytes[6] <= b'7'
}

#[inline]
fn is_hex_digit(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

// ===========================================================================
// Timing Utilities
// ===========================================================================

/// A simple stopwatch for benchmark timing.
///
/// Records the start time on creation and provides elapsed time queries.
///
/// # Validation
///
/// - **VAL-INFRA-013**: Benchmark timing utilities (elapsed calculation, formatting)
#[derive(Debug, Clone)]
pub struct BenchTimer {
    start: Instant,
    label: String,
}

impl BenchTimer {
    /// Create a new timer with the given label, starting immediately.
    pub fn start(label: &str) -> Self {
        Self {
            start: Instant::now(),
            label: label.to_string(),
        }
    }

    /// Create a new timer with label "benchmark".
    pub fn start_default() -> Self {
        Self::start("benchmark")
    }

    /// Return the elapsed time since start.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Return elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> u128 {
        self.elapsed().as_millis()
    }

    /// Return elapsed time in seconds (floating point).
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    /// Return the timer's label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Reset the timer, starting from now.
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

/// Format a `Duration` as a human-readable string.
///
/// Returns milliseconds for durations under 1 second, otherwise seconds with
/// 3 decimal places.
///
/// ```
/// use rusty_stack::installers::common::benchmark_common::format_elapsed;
/// use std::time::Duration;
///
/// assert_eq!(format_elapsed(Duration::from_millis(500)), "500ms");
/// assert_eq!(format_elapsed(Duration::from_millis(1500)), "1.500s");
/// ```
pub fn format_elapsed(duration: Duration) -> String {
    let ms = duration.as_millis();
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.3}s", duration.as_secs_f64())
    }
}

// ===========================================================================
// BenchmarkResult Struct
// ===========================================================================

/// A single benchmark measurement result.
///
/// Contains the benchmark name, numeric value, unit, timestamp, and optional
/// GPU identifier. Serializes to/from JSON compatible with `benchmark_logs.rs`
/// parsing logic.
///
/// # Validation
///
/// - **VAL-INFRA-013**: BenchmarkResult struct with name, value, unit,
///   timestamp, gpu_id; JSON serialization round-trip with benchmark_logs.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkResult {
    /// Human-readable benchmark name (e.g., "memory_bandwidth").
    pub name: String,
    /// Numeric benchmark value.
    pub value: f64,
    /// Unit of measurement (e.g., "GB/s", "ms", "TFLOPS").
    pub unit: String,
    /// ISO 8601 timestamp when the benchmark was recorded.
    pub timestamp: String,
    /// PCI bus ID of the GPU used (e.g., "01:00.0"), or empty if CPU-only.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub gpu_id: String,
    /// Whether the benchmark succeeded.
    #[serde(default = "default_true")]
    pub success: bool,
    /// Optional error message (set when `success` is false).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn default_true() -> bool {
    true
}

impl BenchmarkResult {
    /// Create a new successful benchmark result with the current timestamp.
    pub fn new(name: &str, value: f64, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            value,
            unit: unit.to_string(),
            timestamp: now_timestamp(),
            gpu_id: String::new(),
            success: true,
            error: None,
        }
    }

    /// Create a new benchmark result with a specific GPU ID.
    pub fn with_gpu(name: &str, value: f64, unit: &str, gpu_id: &str) -> Self {
        Self {
            gpu_id: gpu_id.to_string(),
            ..Self::new(name, value, unit)
        }
    }

    /// Create a failed benchmark result.
    pub fn failed(name: &str, error: &str) -> Self {
        Self {
            name: name.to_string(),
            value: 0.0,
            unit: String::new(),
            timestamp: now_timestamp(),
            gpu_id: String::new(),
            success: false,
            error: Some(error.to_string()),
        }
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.success {
            write!(f, "{}: {:.3} {}", self.name, self.value, self.unit)
        } else {
            write!(
                f,
                "{}: FAILED ({})",
                self.name,
                self.error.as_deref().unwrap_or("unknown error")
            )
        }
    }
}

/// A collection of benchmark results, compatible with the JSON schema used
/// by `benchmark_logs.rs`.
///
/// # Validation
///
/// - **VAL-INFRA-013**: JSON serialization round-trip with benchmark_logs.rs types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkResults {
    /// The benchmark suite name.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Individual benchmark results.
    pub results: Vec<BenchmarkResult>,
    /// Optional metadata (GPU info, ROCm version, etc.).
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
}

impl BenchmarkResults {
    /// Create an empty result collection.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            results: Vec::new(),
            metadata: serde_json::Value::Null,
        }
    }

    /// Add a result to the collection.
    pub fn add(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Number of results in the collection.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

// ===========================================================================
// Timestamp Utilities
// ===========================================================================

/// Get the current timestamp in `YYYY-MM-DD HH:MM:SS` format.
///
/// ```
/// use rusty_stack::installers::common::benchmark_common::now_timestamp;
///
/// let ts = now_timestamp();
/// assert!(ts.len() == 19);
/// assert!(ts.starts_with("20"));
/// assert_eq!(&ts[4..5], "-");
/// assert_eq!(&ts[13..14], ":");
/// ```
///
/// # Validation
///
/// - **VAL-INFRA-014**: `[YYYY-MM-DD HH:MM:SS]` prefix format
pub fn now_timestamp() -> String {
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs() as i64;

    // Simple date/time calculation from Unix epoch
    let (year, month, day, hour, minute, second) = unix_to_datetime(secs);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month, day, hour, minute, second
    )
}

/// Convert Unix timestamp to (year, month, day, hour, minute, second).
///
/// Uses a simplified algorithm that avoids external dependencies.
fn unix_to_datetime(secs: i64) -> (i64, i64, i64, i64, i64, i64) {
    let mut remaining = secs;
    let second = remaining % 60;
    remaining /= 60;
    let minute = remaining % 60;
    remaining /= 60;
    let hour = remaining % 24;
    let mut days = remaining / 24;

    // Unix epoch starts at 1970-01-01
    let mut year = 1970i64;
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let month_days: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1i64;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    let day = days + 1;

    (year, month, day, hour, minute, second)
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Format a timestamp string for use in log lines.
///
/// Returns `[YYYY-MM-DD HH:MM:SS]` wrapped in brackets.
pub fn format_timestamp_prefix() -> String {
    format!("[{}] ", now_timestamp())
}

// ===========================================================================
// Timestamped Logging
// ===========================================================================

/// Benchmark logger that writes timestamped lines to stderr and optionally
/// to a log file (tee behavior).
///
/// # Validation
///
/// - **VAL-INFRA-014**: Timestamped logging `[YYYY-MM-DD HH:MM:SS]` prefix,
///   supports stderr + file output (tee behavior)
#[derive(Debug)]
pub struct BenchLogger {
    /// Optional log file path for tee output.
    log_file: Option<PathBuf>,
    /// Whether color output is enabled.
    color_enabled: bool,
}

/// Color constants for benchmark logging.
struct BenchColors {
    red: &'static str,
    green: &'static str,
    yellow: &'static str,
    blue: &'static str,
    reset: &'static str,
}

const COLOR: BenchColors = BenchColors {
    red: "\x1b[0;31m",
    green: "\x1b[0;32m",
    yellow: "\x1b[0;33m",
    blue: "\x1b[0;34m",
    reset: "\x1b[0m",
};

const NO_COLOR: BenchColors = BenchColors {
    red: "",
    green: "",
    yellow: "",
    blue: "",
    reset: "",
};

impl BenchLogger {
    /// Create a new logger with no file output.
    pub fn new() -> Self {
        Self {
            log_file: None,
            color_enabled: color_output_enabled(),
        }
    }

    /// Create a new logger that also writes to the given file path.
    ///
    /// The file is opened in append mode; if it doesn't exist, it is created.
    pub fn with_file<P: AsRef<Path>>(path: P) -> Self {
        Self {
            log_file: Some(path.as_ref().to_path_buf()),
            color_enabled: color_output_enabled(),
        }
    }

    /// Set the log file path (replaces any existing one).
    pub fn set_log_file<P: AsRef<Path>>(&mut self, path: P) {
        self.log_file = Some(path.as_ref().to_path_buf());
    }

    /// Clear the log file (stderr-only mode).
    pub fn clear_log_file(&mut self) {
        self.log_file = None;
    }

    /// Get the log file path, if set.
    pub fn log_file_path(&self) -> Option<&Path> {
        self.log_file.as_deref()
    }

    /// Log a raw message with timestamp prefix.
    ///
    /// Writes to stderr and, if configured, to the log file (tee behavior).
    pub fn log(&self, message: &str) {
        let line = format!("[{}] {}", now_timestamp(), message);
        eprintln!("{}", line);

        if let Some(ref path) = self.log_file {
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                // Strip ANSI color codes for file output
                let stripped = strip_ansi_escapes(&line);
                let _ = writeln!(file, "{}", stripped);
            }
        }
    }

    /// Log an info-level message.
    pub fn info(&self, message: &str) {
        let c = self.colors();
        self.log(&format!("{}INFO{} {}", c.blue, c.reset, message));
    }

    /// Log a warning-level message.
    pub fn warn(&self, message: &str) {
        let c = self.colors();
        self.log(&format!("{}WARN{} {}", c.yellow, c.reset, message));
    }

    /// Log an error-level message.
    pub fn error(&self, message: &str) {
        let c = self.colors();
        self.log(&format!("{}ERROR{} {}", c.red, c.reset, message));
    }

    /// Log a success/OK message.
    pub fn success(&self, message: &str) {
        let c = self.colors();
        self.log(&format!("{}OK{} {}", c.green, c.reset, message));
    }

    fn colors(&self) -> &'static BenchColors {
        if self.color_enabled {
            &COLOR
        } else {
            &NO_COLOR
        }
    }
}

impl Default for BenchLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Determine whether color output should be enabled.
///
/// Returns `false` if `NO_COLOR` is set or stderr is not a TTY.
fn color_output_enabled() -> bool {
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    use std::io::IsTerminal;
    std::io::stderr().is_terminal()
}

/// Strip ANSI escape sequences from a string (for file output).
fn strip_ansi_escapes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // Skip the escape sequence
            if chars.peek() == Some(&'[') {
                chars.next();
                // Consume sequence: digits and semicolons followed by a letter
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            result.push(ch);
        }
    }
    result
}

// ===========================================================================
// Log Directory Resolution
// ===========================================================================

/// Resolve the benchmark log directory.
///
/// Mirrors `benchmark_resolve_log_dir()` from the shell script:
/// - Prefers `MLSTACK_LOG_DIR` environment variable
/// - Falls back to `$HOME/.rusty-stack/logs`
/// - Final fallback to `${TMPDIR:-/tmp}/rusty-stack/logs`
///
/// Creates the directory if it doesn't exist.
pub fn resolve_log_dir() -> PathBuf {
    if let Ok(env_dir) = std::env::var("MLSTACK_LOG_DIR") {
        let trimmed = env_dir.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(trimmed);
            if ensure_dir(&path) {
                return path;
            }
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(trimmed).join(".rusty-stack").join("logs");
            if ensure_dir(&path) {
                return path;
            }
        }
    }

    let tmp = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    let path = PathBuf::from(tmp.trim()).join("rusty-stack").join("logs");
    let _ = std::fs::create_dir_all(&path);
    path
}

/// Ensure a directory exists and is writable.
fn ensure_dir(path: &Path) -> bool {
    if std::fs::create_dir_all(path).is_ok() {
        // Check writability by trying to create a temp file
        let probe = path.join(".bench_write_probe");
        if let Ok(f) = File::create(&probe) {
            let _ = f;
            let _ = std::fs::remove_file(&probe);
            return true;
        }
    }
    false
}

// ===========================================================================
// Dry-Run Support
// ===========================================================================

/// Check if dry-run mode is enabled via environment variable.
pub fn is_dry_run() -> bool {
    std::env::var("DRY_RUN")
        .map(|v| v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // ----- PCI Bus ID Normalization Tests -----

    #[test]
    fn test_pci_normalize_strips_domain_prefix() {
        assert_eq!(
            normalize_pci_bus_id("0000:01:00.0"),
            Ok("01:00.0".to_string())
        );
    }

    #[test]
    fn test_pci_normalize_already_short() {
        assert_eq!(normalize_pci_bus_id("01:00.0"), Ok("01:00.0".to_string()));
    }

    #[test]
    fn test_pci_normalize_uppercase() {
        assert_eq!(
            normalize_pci_bus_id("0000:0A:1B.2"),
            Ok("0a:1b.2".to_string())
        );
    }

    #[test]
    fn test_pci_normalize_mixed_case() {
        assert_eq!(normalize_pci_bus_id("0A:1B.3"), Ok("0a:1b.3".to_string()));
    }

    #[test]
    fn test_pci_normalize_with_whitespace() {
        assert_eq!(
            normalize_pci_bus_id("  0000:03:00.0  "),
            Ok("03:00.0".to_string())
        );
    }

    #[test]
    fn test_pci_normalize_with_trailing_text() {
        // Takes only the first token
        assert_eq!(
            normalize_pci_bus_id("0000:05:00.0 some extra text"),
            Ok("05:00.0".to_string())
        );
    }

    #[test]
    fn test_pci_normalize_full_domain_format() {
        assert_eq!(
            normalize_pci_bus_id("0000:0f:00.0"),
            Ok("0f:00.0".to_string())
        );
    }

    #[test]
    fn test_pci_normalize_various_valid() {
        assert_eq!(normalize_pci_bus_id("00:00.0"), Ok("00:00.0".to_string()));
        assert_eq!(normalize_pci_bus_id("ff:ff.7"), Ok("ff:ff.7".to_string()));
        assert_eq!(normalize_pci_bus_id("ab:cd.0"), Ok("ab:cd.0".to_string()));
    }

    #[test]
    fn test_pci_normalize_invalid_empty() {
        assert!(normalize_pci_bus_id("").is_err());
    }

    #[test]
    fn test_pci_normalize_invalid_garbage() {
        assert!(normalize_pci_bus_id("not-a-bus-id").is_err());
    }

    #[test]
    fn test_pci_normalize_invalid_function_too_high() {
        // Function number must be 0-7
        assert!(normalize_pci_bus_id("01:00.8").is_err());
    }

    #[test]
    fn test_pci_normalize_invalid_missing_parts() {
        assert!(normalize_pci_bus_id("01:00").is_err());
        assert!(normalize_pci_bus_id("01").is_err());
    }

    // ----- Timing Tests -----

    #[test]
    fn test_timer_elapsed_increases() {
        let timer = BenchTimer::start("test");
        let e1 = timer.elapsed_ms();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let e2 = timer.elapsed_ms();
        assert!(e2 >= e1, "elapsed should be non-decreasing");
    }

    #[test]
    fn test_timer_label() {
        let timer = BenchTimer::start("my_bench");
        assert_eq!(timer.label(), "my_bench");
    }

    #[test]
    fn test_timer_default_label() {
        let timer = BenchTimer::start_default();
        assert_eq!(timer.label(), "benchmark");
    }

    #[test]
    fn test_timer_reset() {
        let mut timer = BenchTimer::start("test");
        std::thread::sleep(std::time::Duration::from_millis(20));
        let before = timer.elapsed_ms();
        timer.reset();
        let after = timer.elapsed_ms();
        assert!(
            after < before,
            "after reset, elapsed should be smaller: after={after}, before={before}"
        );
    }

    #[test]
    fn test_format_elapsed_milliseconds() {
        assert_eq!(format_elapsed(Duration::from_millis(0)), "0ms");
        assert_eq!(format_elapsed(Duration::from_millis(500)), "500ms");
        assert_eq!(format_elapsed(Duration::from_millis(999)), "999ms");
    }

    #[test]
    fn test_format_elapsed_seconds() {
        assert_eq!(format_elapsed(Duration::from_millis(1000)), "1.000s");
        assert_eq!(format_elapsed(Duration::from_millis(1500)), "1.500s");
        assert_eq!(format_elapsed(Duration::from_secs(60)), "60.000s");
    }

    // ----- BenchmarkResult Tests -----

    #[test]
    fn test_benchmark_result_new() {
        let r = BenchmarkResult::new("memory_bandwidth", 512.3, "GB/s");
        assert_eq!(r.name, "memory_bandwidth");
        assert!((r.value - 512.3).abs() < f64::EPSILON);
        assert_eq!(r.unit, "GB/s");
        assert!(r.success);
        assert!(r.error.is_none());
        assert!(r.gpu_id.is_empty());
        assert!(!r.timestamp.is_empty());
    }

    #[test]
    fn test_benchmark_result_with_gpu() {
        let r = BenchmarkResult::with_gpu("flops", 10.5, "TFLOPS", "01:00.0");
        assert_eq!(r.gpu_id, "01:00.0");
    }

    #[test]
    fn test_benchmark_result_failed() {
        let r = BenchmarkResult::failed("rocm_bench", "ROCm not found");
        assert_eq!(r.name, "rocm_bench");
        assert!(!r.success);
        assert_eq!(r.error.as_deref(), Some("ROCm not found"));
        assert_eq!(r.value, 0.0);
    }

    #[test]
    fn test_benchmark_result_json_roundtrip() {
        let original = BenchmarkResult::with_gpu("bandwidth", 256.7, "GB/s", "03:00.0");
        let json = original.to_json().expect("serialization should succeed");
        let restored = BenchmarkResult::from_json(&json).expect("deserialization should succeed");

        assert_eq!(original, restored);
    }

    #[test]
    fn test_benchmark_result_json_roundtrip_failed() {
        let original = BenchmarkResult::failed("flops", "GPU error");
        let json = original.to_json().expect("serialization should succeed");
        let restored = BenchmarkResult::from_json(&json).expect("deserialization should succeed");

        assert_eq!(original, restored);
    }

    #[test]
    fn test_benchmark_result_json_has_expected_fields() {
        let r = BenchmarkResult::new("test_bench", 42.0, "ms");
        let json = r.to_json().expect("serialization should succeed");
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(value.get("name").is_some());
        assert!(value.get("value").is_some());
        assert!(value.get("unit").is_some());
        assert!(value.get("timestamp").is_some());
        assert!(value.get("success").is_some());
        // gpu_id should be skipped when empty
        assert!(value.get("gpu_id").is_none());
        // error should be skipped when None
        assert!(value.get("error").is_none());
    }

    #[test]
    fn test_benchmark_result_json_includes_gpu_id() {
        let r = BenchmarkResult::with_gpu("test", 1.0, "s", "01:00.0");
        let json = r.to_json().expect("serialization should succeed");
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["gpu_id"].as_str(), Some("01:00.0"));
    }

    #[test]
    fn test_benchmark_result_display_success() {
        let r = BenchmarkResult::new("bandwidth", 512.3, "GB/s");
        let display = format!("{r}");
        assert!(display.contains("bandwidth"));
        assert!(display.contains("512"));
        assert!(display.contains("GB/s"));
    }

    #[test]
    fn test_benchmark_result_display_failed() {
        let r = BenchmarkResult::failed("test", "error msg");
        let display = format!("{r}");
        assert!(display.contains("FAILED"));
        assert!(display.contains("error msg"));
    }

    // ----- BenchmarkResults Collection Tests -----

    #[test]
    fn test_benchmark_results_collection() {
        let mut results = BenchmarkResults::new("gpu_suite");
        assert!(results.is_empty());

        results.add(BenchmarkResult::new("bench1", 1.0, "ms"));
        results.add(BenchmarkResult::new("bench2", 2.0, "ms"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_benchmark_results_json_roundtrip() {
        let mut results = BenchmarkResults::new("gpu_suite");
        results.add(BenchmarkResult::with_gpu("bw", 512.0, "GB/s", "01:00.0"));
        results.add(BenchmarkResult::new("latency", 0.5, "ms"));
        results.metadata = serde_json::json!({"rocm_version": "7.2.1"});

        let json = results.to_json().expect("serialization should succeed");
        let restored = BenchmarkResults::from_json(&json).expect("deserialization should succeed");

        assert_eq!(results, restored);
    }

    #[test]
    fn test_benchmark_results_json_compatible_with_benchmark_logs() {
        // Verify that the JSON output can be parsed by benchmark_logs::extract_benchmark_json_value
        let mut results = BenchmarkResults::new("test_suite");
        results.add(BenchmarkResult::new("mem_bw", 256.0, "GB/s"));

        let json = results.to_json().unwrap();
        let parsed = crate::benchmark_logs::extract_benchmark_json_value(&json);
        assert!(
            parsed.is_some(),
            "benchmark_logs should be able to parse our JSON"
        );

        let value = parsed.unwrap();
        assert!(value.get("results").is_some());
    }

    // ----- Timestamp Tests -----

    #[test]
    fn test_now_timestamp_format() {
        let ts = now_timestamp();
        // Format: YYYY-MM-DD HH:MM:SS (19 chars)
        assert_eq!(ts.len(), 19, "timestamp should be 19 chars: {ts}");
        // Year should start with 20xx
        assert!(ts.starts_with("20"), "year should be 20xx: {ts}");
        // Separators
        assert_eq!(&ts[4..5], "-", "separator at pos 4 should be -");
        assert_eq!(&ts[7..8], "-", "separator at pos 7 should be -");
        assert_eq!(&ts[10..11], " ", "separator at pos 10 should be space");
        assert_eq!(&ts[13..14], ":", "separator at pos 13 should be :");
        assert_eq!(&ts[16..17], ":", "separator at pos 16 should be :");
    }

    #[test]
    fn test_format_timestamp_prefix() {
        let prefix = format_timestamp_prefix();
        assert!(prefix.starts_with('['), "prefix should start with [");
        assert!(prefix.ends_with("] "), "prefix should end with ] ");
        // Length: [YYYY-MM-DD HH:MM:SS] + space = 22
        assert_eq!(prefix.len(), 22, "prefix should be 22 chars: {prefix}");
    }

    // We use a simple manual validation instead of regex
    fn validate_timestamp_format(ts: &str) -> bool {
        if ts.len() != 19 {
            return false;
        }
        let bytes = ts.as_bytes();
        // YYYY-MM-DD HH:MM:SS
        bytes[0].is_ascii_digit()
            && bytes[1].is_ascii_digit()
            && bytes[2].is_ascii_digit()
            && bytes[3].is_ascii_digit()
            && bytes[4] == b'-'
            && bytes[5].is_ascii_digit()
            && bytes[6].is_ascii_digit()
            && bytes[7] == b'-'
            && bytes[8].is_ascii_digit()
            && bytes[9].is_ascii_digit()
            && bytes[10] == b' '
            && bytes[11].is_ascii_digit()
            && bytes[12].is_ascii_digit()
            && bytes[13] == b':'
            && bytes[14].is_ascii_digit()
            && bytes[15].is_ascii_digit()
            && bytes[16] == b':'
            && bytes[17].is_ascii_digit()
            && bytes[18].is_ascii_digit()
    }

    #[test]
    fn test_timestamp_format_validation() {
        let ts = now_timestamp();
        assert!(
            validate_timestamp_format(&ts),
            "timestamp '{ts}' should validate"
        );
    }

    // ----- Logging Tests -----

    #[test]
    fn test_bench_logger_log_to_file() {
        let tmp = TempDir::new().expect("temp dir");
        let log_path = tmp.path().join("bench.log");

        let logger = BenchLogger::with_file(&log_path);
        logger.info("test message");

        let contents = fs::read_to_string(&log_path).expect("read log file");
        assert!(contents.contains("INFO"), "log should contain INFO level");
        assert!(
            contents.contains("test message"),
            "log should contain message"
        );
        // File output should not contain ANSI escape codes
        assert!(
            !contents.contains('\x1b'),
            "file should not have ANSI escapes"
        );
    }

    #[test]
    fn test_bench_logger_tee_behavior() {
        let tmp = TempDir::new().expect("temp dir");
        let log_path = tmp.path().join("tee.log");

        let logger = BenchLogger::with_file(&log_path);
        logger.warn("warning message");
        logger.error("error message");
        logger.success("success message");

        let contents = fs::read_to_string(&log_path).expect("read log file");
        assert!(contents.contains("WARN"), "should have WARN");
        assert!(contents.contains("ERROR"), "should have ERROR");
        assert!(contents.contains("OK"), "should have OK");
        assert!(contents.contains("warning message"));
        assert!(contents.contains("error message"));
        assert!(contents.contains("success message"));
    }

    #[test]
    fn test_bench_logger_no_file() {
        let logger = BenchLogger::new();
        assert!(logger.log_file_path().is_none());
        // Should not panic when logging without a file
        logger.info("no file test");
    }

    #[test]
    fn test_bench_logger_set_and_clear_file() {
        let mut logger = BenchLogger::new();
        assert!(logger.log_file_path().is_none());

        let tmp = TempDir::new().expect("temp dir");
        let log_path = tmp.path().join("dynamic.log");
        logger.set_log_file(&log_path);
        assert!(logger.log_file_path().is_some());

        logger.info("written to dynamic file");
        let contents = fs::read_to_string(&log_path).expect("read log");
        assert!(contents.contains("written to dynamic file"));

        logger.clear_log_file();
        assert!(logger.log_file_path().is_none());
    }

    #[test]
    fn test_bench_logger_timestamp_in_file() {
        let tmp = TempDir::new().expect("temp dir");
        let log_path = tmp.path().join("ts.log");

        let logger = BenchLogger::with_file(&log_path);
        logger.log("plain message");

        let contents = fs::read_to_string(&log_path).expect("read log");
        // Should contain [YYYY-MM-DD HH:MM:SS] prefix
        assert!(
            validate_timestamp_prefix(&contents),
            "log line should have timestamp prefix: {contents}"
        );
    }

    fn validate_timestamp_prefix(contents: &str) -> bool {
        // Look for [YYYY-MM-DD HH:MM:SS] pattern
        for (i, ch) in contents.char_indices() {
            if ch == '[' && contents.len() > i + 20 {
                let candidate = &contents[i + 1..i + 20];
                if validate_timestamp_format(candidate)
                    && contents.as_bytes().get(i + 20) == Some(&b']')
                {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn test_bench_log_lines_have_timestamps() {
        let tmp = TempDir::new().expect("temp dir");
        let log_path = tmp.path().join("lines.log");

        let logger = BenchLogger::with_file(&log_path);
        logger.info("line1");
        logger.warn("line2");
        logger.error("line3");

        let contents = fs::read_to_string(&log_path).expect("read log");
        for line in contents.lines() {
            assert!(
                line.contains("INFO") || line.contains("WARN") || line.contains("ERROR"),
                "each line should have a level marker: {line}"
            );
        }
    }

    // ----- ANSI Stripping Tests -----

    #[test]
    fn test_strip_ansi_escapes() {
        let colored = "\x1b[0;31mERROR\x1b[0m something";
        let stripped = strip_ansi_escapes(colored);
        assert_eq!(stripped, "ERROR something");
    }

    #[test]
    fn test_strip_ansi_no_escapes() {
        let plain = "plain text";
        assert_eq!(strip_ansi_escapes(plain), "plain text");
    }

    // ----- Log Directory Tests -----

    #[test]
    fn test_resolve_log_dir_creates_dir() {
        let tmp = TempDir::new().expect("temp dir");
        let log_dir = tmp.path().join("test_logs");

        std::env::set_var("MLSTACK_LOG_DIR", log_dir.to_str().unwrap());
        let result = resolve_log_dir();
        std::env::remove_var("MLSTACK_LOG_DIR");

        assert!(result.ends_with("test_logs"));
        assert!(log_dir.exists());
    }

    // ----- Dry-Run Tests -----

    #[test]
    fn test_dry_run_false_by_default() {
        std::env::remove_var("DRY_RUN");
        assert!(!is_dry_run());
    }

    #[test]
    fn test_dry_run_true_when_set() {
        std::env::set_var("DRY_RUN", "true");
        assert!(is_dry_run());
        std::env::remove_var("DRY_RUN");
    }

    #[test]
    fn test_dry_run_case_insensitive() {
        // Ensure clean state
        std::env::remove_var("DRY_RUN");

        std::env::set_var("DRY_RUN", "TRUE");
        assert!(is_dry_run());
        std::env::remove_var("DRY_RUN");

        std::env::set_var("DRY_RUN", "True");
        assert!(is_dry_run());
        std::env::remove_var("DRY_RUN");
    }

    #[test]
    fn test_dry_run_false_when_other_value() {
        std::env::set_var("DRY_RUN", "false");
        assert!(!is_dry_run());
        std::env::remove_var("DRY_RUN");

        std::env::set_var("DRY_RUN", "0");
        assert!(!is_dry_run());
        std::env::remove_var("DRY_RUN");
    }

    // ----- Integration: BenchmarkResult + benchmark_logs compatibility -----

    #[test]
    fn test_single_result_benchmark_logs_compat() {
        let r = BenchmarkResult::new("test_bench", 42.0, "ms");
        let json = r.to_json().unwrap();
        // benchmark_logs should be able to parse this
        let parsed = crate::benchmark_logs::extract_benchmark_json_value(&json);
        assert!(parsed.is_some());
        let val = parsed.unwrap();
        assert_eq!(val["name"].as_str(), Some("test_bench"));
        assert!((val["value"].as_f64().unwrap() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_results_collection_benchmark_logs_compat() {
        let mut results = BenchmarkResults::new("full_suite");
        results.add(BenchmarkResult::new("bw", 100.0, "GB/s"));
        results.add(BenchmarkResult::failed("broken", "oops"));

        let json = results.to_json().unwrap();
        let parsed = crate::benchmark_logs::extract_benchmark_json_value(&json);
        assert!(parsed.is_some());
        let val = parsed.unwrap();
        assert!(val.get("results").is_some());
    }
}
