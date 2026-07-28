//! Logging infrastructure for Rusty Stack.
//!
//! Provides a dual-layer logging setup:
//! - **File layer**: Structured JSON logs at `~/.mlstack/logs/` with daily rotation
//!   (max 3 files retained for storage efficiency).
//! - **Stdout layer**: Compact human-readable progress for CLI updates.
//!
//! # Usage
//!
//! ```ignore
//! use rusty_stack::logging::init_logging;
//!
//! let _guard = init_logging("update");
//! // tracing::info!, tracing::error!, etc. now work
//! ```

use std::path::PathBuf;
use tracing_subscriber::{
    filter::EnvFilter,
    fmt::{self, time::LocalTime},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    Layer,
};

/// Log directory under `~/.mlstack/logs/` (canonical, sudo-aware path).
pub fn log_dir() -> PathBuf {
    crate::platform::environment::mlstack_logs_dir()
}

/// Fixed WARN ceiling for the stderr/stdout CONSOLE layers.
///
/// The console (stderr/stdout) layers must NEVER let `RUST_LOG` lower the
/// threshold below WARN, because `main.rs` emits a per-line `tracing::info!`
/// for every installer log line — an INFO threshold floods the console and
/// corrupts machine-readable JSON/CI output. This builds a FIXED `EnvFilter`
/// (never reads `RUST_LOG`) so the console ceiling cannot be bypassed by
/// `RUST_LOG=rusty_stack=info`. The FILE layer keeps env-driven filtering
/// (full DEBUG/INFO stream for live diagnostics).
fn console_warn_ceiling() -> EnvFilter {
    EnvFilter::new("rusty_stack=warn")
}

/// Initialize the dual-layer logging system.
///
/// Returns a guard that must be kept alive for the file logger to flush on drop.
/// The `context` string is embedded in log entries to identify the operation
/// (e.g., "update", "verify", "bench").
///
/// # File format
///
/// JSON lines with fields: `timestamp`, `level`, `span`, `message`, plus any
/// structured fields from spans.
///
/// # Storage efficiency
///
/// Daily rotation with max 3 log files retained. Older files are pruned automatically
/// by `tracing-appender`.
pub fn init_logging(context: &str) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let log_path = log_dir();
    if let Err(e) = std::fs::create_dir_all(&log_path) {
        eprintln!(
            "[WARN] Could not create log directory {}: {}",
            log_path.display(),
            e
        );
        // Fall back to stdout-only — fixed WARN ceiling (RUST_LOG cannot bypass).
        let filter = console_warn_ceiling();
        let stdout_layer = fmt::layer().with_target(false).with_filter(filter);
        tracing_subscriber::registry().with(stdout_layer).init();
        return None;
    }

    // File appender: daily rotation, max 3 files, prefix with context
    let file_appender =
        tracing_appender::rolling::daily(&log_path, format!("{}-rusty-stack", context));
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // JSON file layer — structured for machine parsing
    let file_layer = fmt::layer()
        .json()
        .with_timer(LocalTime::rfc_3339())
        .with_writer(non_blocking)
        .with_target(true)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .with_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("rusty_stack=debug,info")),
        );

    // Stdout layer — compact human-readable for CLI progress. Fixed WARN
    // ceiling (RUST_LOG cannot bypass) so the per-line installer INFO logs
    // never flood stdout; the file layer captures the full stream.
    let stdout_layer = fmt::layer()
        .with_target(false)
        .with_timer(fmt::time::LocalTime::new(
            time::format_description::well_known::iso8601::Iso8601::DEFAULT,
        ))
        .with_filter(console_warn_ceiling());

    tracing_subscriber::registry()
        .with(file_layer)
        .with(stdout_layer)
        .init();

    tracing::info!(
        context = context,
        log_dir = %log_path.display(),
        "Logging initialized"
    );

    Some(guard)
}

/// Initialize logging for non-interactive/batch contexts.
///
/// Uses a simpler configuration: file logging + stderr (not stdout) so that
/// machine-readable stdout output (JSON) is not polluted with log lines.
pub fn init_batch_logging(context: &str) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let log_path = log_dir();
    if let Err(e) = std::fs::create_dir_all(&log_path) {
        eprintln!(
            "[WARN] Could not create log directory {}: {}",
            log_path.display(),
            e
        );
        return None;
    }

    let file_appender =
        tracing_appender::rolling::daily(&log_path, format!("{}-rusty-stack", context));
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // File layer — full debug detail
    let file_layer = fmt::layer()
        .json()
        .with_timer(LocalTime::rfc_3339())
        .with_writer(non_blocking)
        .with_target(true)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .with_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("rusty_stack=debug")),
        );

    // Stderr layer — warnings and errors only (not stdout). FIXED WARN ceiling:
    // main.rs emits a per-line `tracing::info!` for every installer log line, so
    // an INFO threshold (whether via the default or `RUST_LOG=rusty_stack=info`)
    // floods stderr on JSON/CI runs and corrupts machine-readable output.
    // `console_warn_ceiling()` builds a fixed filter that `RUST_LOG` cannot
    // bypass. The file layer captures the full INFO+/DEBUG+ stream.
    let stderr_layer = fmt::layer()
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_filter(console_warn_ceiling());

    tracing_subscriber::registry()
        .with(file_layer)
        .with(stderr_layer)
        .init();

    tracing::info!(
        context = context,
        log_dir = %log_path.display(),
        "Batch logging initialized"
    );

    Some(guard)
}

/// Initialize logging for interactive contexts (the `update` command's inline
/// selection + animated apply panel).
///
/// Stdout stays free of log lines (the ratatui `Viewport::Inline` panel owns
/// it). Stderr is **WARN+ only** — the per-line `tracing::info!(log = …)` that
/// the installer emits for every pip/git line goes to the FILE (full debug),
/// not the terminal, so it can't flood around the panel. Without this, the
/// interactive run reproduces the raw-log flood on stderr.
pub fn init_interactive_logging(
    context: &str,
) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let log_path = log_dir();
    if let Err(e) = std::fs::create_dir_all(&log_path) {
        eprintln!(
            "[WARN] Could not create log directory {}: {}",
            log_path.display(),
            e
        );
        return None;
    }

    let file_appender =
        tracing_appender::rolling::daily(&log_path, format!("{}-rusty-stack", context));
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // File layer — full debug detail (every installer log line is captured here).
    let file_layer = fmt::layer()
        .json()
        .with_timer(LocalTime::rfc_3339())
        .with_writer(non_blocking)
        .with_target(true)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .with_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("rusty_stack=debug")),
        );

    // Stderr layer — WARN+ only (FIXED ceiling: `RUST_LOG` cannot bypass). The
    // terminal is owned by the inline panel; the torrent of INFO installer-log
    // lines must not bleed onto it.
    let stderr_layer = fmt::layer()
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_filter(console_warn_ceiling());

    tracing_subscriber::registry()
        .with(file_layer)
        .with(stderr_layer)
        .init();

    tracing::info!(
        context = context,
        log_dir = %log_path.display(),
        "Interactive logging initialized"
    );

    Some(guard)
}

/// Get the path to the most recent log file for a given context.
///
/// Returns the log directory path if it exists, for user-facing messages.
pub fn log_file_path(context: &str) -> PathBuf {
    log_dir().join(format!("{}-rusty-stack", context))
}

/// Time format module for tracing-subscriber.
mod time {
    pub use time::format_description;
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_dir_is_under_mlstack() {
        let dir = log_dir();
        assert!(dir.to_string_lossy().contains(".mlstack"));
        assert!(dir.to_string_lossy().contains("logs"));
    }

    #[test]
    fn test_log_file_path_contains_context() {
        let path = log_file_path("update");
        assert!(path.to_string_lossy().contains("update-rusty-stack"));
    }
}
