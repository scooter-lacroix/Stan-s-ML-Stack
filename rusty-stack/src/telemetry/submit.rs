//! HTTPS telemetry submission client.
//!
//! Submits anonymous telemetry payloads exclusively over HTTPS.
//! HTTP URLs are rejected before any network access occurs.
//!
//! # Design
//!
//! - **HTTPS-only**: HTTP URLs are rejected at validation time, before any
//!   network connection is made.
//! - **Fire-and-forget**: Submission runs in a background task via
//!   [`submit_fire_and_forget`] and does not block the main pipeline.
//! - **Thank-you confirmation**: On successful submission (HTTP 200), a
//!   thank-you message is displayed to the user.
//!
//! # Validation Contract
//!
//! - **VAL-TELE-008**: Submission exclusively over HTTPS, HTTP rejected.
//! - **VAL-TELE-009**: Thank-you confirmation displayed after successful
//!   submission.

use crate::core::telemetry_types::TelemetryPayload;
use crate::telemetry::payload::serialize_payload;
use anyhow::{Context, Result};
use std::sync::mpsc;

/// Default telemetry submission endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://telemetry.rusty-stack.example.com/api/v1/submit";

/// Thank-you message displayed after successful submission.
pub const THANK_YOU_MESSAGE: &str =
    "✓ Thank you! Your anonymous hardware data has been submitted successfully.";

/// Error message when submission fails (non-blocking).
const SUBMISSION_ERROR_PREFIX: &str = "Telemetry submission failed:";

/// Possible outcomes of a telemetry submission attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitOutcome {
    /// Submission succeeded (HTTP 200 OK).
    Success,
    /// Submission failed with an error message.
    Failed(String),
    /// URL was rejected because it is not HTTPS.
    HttpRejected,
}

/// Validate that a URL uses the HTTPS scheme.
///
/// Returns `Ok(())` if the URL starts with `https://`, or an error
/// describing why it was rejected. This check happens **before** any
/// network access.
pub fn validate_https_url(url: &str) -> Result<()> {
    let trimmed = url.trim();

    if trimmed.starts_with("https://") {
        Ok(())
    } else if trimmed.starts_with("http://") {
        anyhow::bail!(
            "HTTP URL rejected — telemetry must be submitted over HTTPS only. URL: {}",
            trimmed
        )
    } else {
        anyhow::bail!(
            "Invalid telemetry URL — must start with https://. URL: {}",
            trimmed
        )
    }
}

/// Submit a telemetry payload to the given endpoint synchronously.
///
/// This function:
/// 1. Validates the URL is HTTPS (rejects HTTP before network access)
/// 2. Serializes and validates the payload
/// 3. Sends the payload via HTTPS POST
/// 4. Returns [`SubmitOutcome::Success`] on HTTP 200
///
/// # Errors
///
/// Returns [`SubmitOutcome::HttpRejected`] for HTTP URLs.
/// Returns [`SubmitOutcome::Failed`] for network or server errors.
pub fn submit_payload(payload: &TelemetryPayload, endpoint: &str) -> Result<SubmitOutcome> {
    // Step 1: Validate HTTPS (VAL-TELE-008)
    if validate_https_url(endpoint).is_err() {
        return Ok(SubmitOutcome::HttpRejected);
    }

    // Step 2: Serialize and validate payload
    let json = serialize_payload(payload).context("Payload validation failed")?;

    // Step 3: Submit via HTTPS
    let rt = tokio::runtime::Runtime::new().context("Failed to create Tokio runtime")?;
    let result = rt.block_on(async {
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                return SubmitOutcome::Failed(format!("Failed to build HTTP client: {}", e));
            }
        };

        let response = client
            .post(endpoint)
            .header("Content-Type", "application/json")
            .body(json)
            .send()
            .await;

        match response {
            Ok(resp) => {
                let status = resp.status();
                if status == reqwest::StatusCode::OK {
                    SubmitOutcome::Success
                } else {
                    SubmitOutcome::Failed(format!("Server returned HTTP {}", status))
                }
            }
            Err(e) => SubmitOutcome::Failed(format!("Network error: {}", e)),
        }
    });

    Ok(result)
}

/// Fire-and-forget telemetry submission.
///
/// Spawns a background thread to submit the payload. Does not block
/// the caller. The submission result is logged but not returned.
///
/// This is the primary submission API for the main pipeline — it adds
/// no measurable latency since it returns immediately.
pub fn submit_fire_and_forget(payload: TelemetryPayload, endpoint: String) -> mpsc::Sender<()> {
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        // Attempt submission
        let outcome = submit_payload(&payload, &endpoint);

        match outcome {
            Ok(SubmitOutcome::Success) => {
                // VAL-TELE-009: Display thank-you confirmation
                println!("{}", THANK_YOU_MESSAGE);
            }
            Ok(SubmitOutcome::Failed(msg)) => {
                // Non-blocking: log error but don't fail pipeline
                eprintln!("{} {}", SUBMISSION_ERROR_PREFIX, msg);
            }
            Ok(SubmitOutcome::HttpRejected) => {
                eprintln!(
                    "{} HTTP URL rejected — telemetry requires HTTPS",
                    SUBMISSION_ERROR_PREFIX
                );
            }
            Err(e) => {
                eprintln!("{} {}", SUBMISSION_ERROR_PREFIX, e);
            }
        }

        // Keep the thread alive until caller drops the sender
        let _ = rx.recv();
    });

    tx
}

/// Submit a telemetry payload and return the outcome directly.
///
/// Unlike [`submit_fire_and_forget`], this function blocks until
/// the submission completes (or fails) and returns the result.
/// Useful for testing and CLI tools that want to report the outcome.
pub fn submit_blocking(payload: &TelemetryPayload, endpoint: &str) -> Result<SubmitOutcome> {
    submit_payload(payload, endpoint)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // VAL-TELE-008: Submission — HTTPS Transport
    // =======================================================================

    #[test]
    fn test_submit_https_url_accepted() {
        let result = validate_https_url("https://telemetry.example.com/submit");
        assert!(result.is_ok(), "HTTPS URL should be accepted: {:?}", result);
    }

    #[test]
    fn test_submit_http_url_rejected_before_network() {
        let result = validate_https_url("http://telemetry.example.com/submit");
        assert!(
            result.is_err(),
            "HTTP URL should be rejected before network access"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("HTTP URL rejected"),
            "Error should mention rejection: {}",
            msg
        );
        assert!(
            msg.contains("HTTPS only"),
            "Error should mention HTTPS requirement: {}",
            msg
        );
    }

    #[test]
    fn test_submit_http_url_returns_http_rejected_outcome() {
        let payload = create_test_payload();
        let outcome = submit_payload(&payload, "http://telemetry.example.com/submit")
            .expect("Should return outcome without error");
        assert_eq!(
            outcome,
            SubmitOutcome::HttpRejected,
            "HTTP URL should produce HttpRejected outcome"
        );
    }

    #[test]
    fn test_submit_invalid_url_rejected() {
        let result = validate_https_url("ftp://telemetry.example.com/submit");
        assert!(result.is_err(), "Non-HTTP URL should be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("must start with https://"),
            "Error should mention https:// requirement: {}",
            msg
        );
    }

    #[test]
    fn test_submit_empty_url_rejected() {
        let result = validate_https_url("");
        assert!(result.is_err(), "Empty URL should be rejected");
    }

    #[test]
    fn test_submit_https_with_path_accepted() {
        let result = validate_https_url("https://example.com/api/v1/submit");
        assert!(result.is_ok(), "HTTPS URL with path should be accepted");
    }

    #[test]
    fn test_submit_https_with_port_accepted() {
        let result = validate_https_url("https://example.com:8443/submit");
        assert!(result.is_ok(), "HTTPS URL with port should be accepted");
    }

    // =======================================================================
    // VAL-TELE-009: Submission — Thank-You Confirmation
    // =======================================================================

    #[test]
    fn test_thank_you_message_is_non_empty() {
        assert!(
            !THANK_YOU_MESSAGE.is_empty(),
            "Thank-you message must not be empty"
        );
        assert!(
            THANK_YOU_MESSAGE.contains("Thank you"),
            "Thank-you message must contain 'Thank you'"
        );
    }

    #[test]
    fn test_submit_outcome_success_display() {
        let outcome = SubmitOutcome::Success;
        assert_eq!(outcome, SubmitOutcome::Success);
    }

    #[test]
    fn test_submit_outcome_failed_contains_message() {
        let outcome = SubmitOutcome::Failed("Server returned HTTP 500".to_string());
        if let SubmitOutcome::Failed(msg) = outcome {
            assert!(msg.contains("500"));
        } else {
            panic!("Expected Failed variant");
        }
    }

    #[test]
    fn test_submit_outcome_http_rejected() {
        let outcome = SubmitOutcome::HttpRejected;
        assert_eq!(outcome, SubmitOutcome::HttpRejected);
    }

    // =======================================================================
    // Fire-and-forget behavior
    // =======================================================================

    #[test]
    fn test_fire_and_forget_returns_immediately() {
        let payload = create_test_payload();
        let endpoint = "https://nonexistent.invalid/submit".to_string();

        // This should return immediately (fire-and-forget)
        let tx = submit_fire_and_forget(payload, endpoint);

        // Drop the sender to let the background thread finish
        drop(tx);

        // If we got here, the function returned immediately
    }

    #[test]
    fn test_fire_and_forget_with_http_url_does_not_panic() {
        let payload = create_test_payload();
        let endpoint = "http://insecure.example.com/submit".to_string();

        let tx = submit_fire_and_forget(payload, endpoint);
        drop(tx);

        // Should not panic even with HTTP URL
    }

    // =======================================================================
    // Default endpoint
    // =======================================================================

    #[test]
    fn test_default_endpoint_is_https() {
        assert!(
            DEFAULT_ENDPOINT.starts_with("https://"),
            "Default endpoint must use HTTPS: {}",
            DEFAULT_ENDPOINT
        );
    }

    // =======================================================================
    // Network failure handling (real HTTPS request to nonexistent host)
    // =======================================================================

    #[test]
    fn test_submit_to_nonexistent_host_returns_failed() {
        let payload = create_test_payload();
        let outcome = submit_payload(&payload, "https://nonexistent.invalid/submit")
            .expect("Should return outcome");

        assert!(
            matches!(outcome, SubmitOutcome::Failed(_)),
            "Should fail for nonexistent host: {:?}",
            outcome
        );
    }

    // =======================================================================
    // Blocking submit
    // =======================================================================

    #[test]
    fn test_submit_blocking_http_rejected() {
        let payload = create_test_payload();
        let outcome =
            submit_blocking(&payload, "http://example.com/submit").expect("Should return outcome");
        assert_eq!(outcome, SubmitOutcome::HttpRejected);
    }

    // =======================================================================
    // Helpers
    // =======================================================================

    fn create_test_payload() -> crate::core::telemetry_types::TelemetryPayload {
        crate::core::telemetry_types::TelemetryPayload::new(
            "gfx1100",
            "AMD Radeon RX 7900 XTX",
            "7.2.1",
            1,
            16,
            54.0,
        )
    }
}
