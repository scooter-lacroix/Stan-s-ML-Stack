//! Telemetry module — opt-in anonymous hardware telemetry and stability benchmarking.
//!
//! This module provides:
//! - **Benchmark**: 180-second minimum stability benchmark collecting GPU metrics
//!   (utilization, VRAM, thermal, throttling, clock speeds) sampled every 10 seconds.
//! - **Payload**: Anonymous hardware-only telemetry payload construction with
//!   strict privacy boundaries (no personal data, bounded to 64 KiB JSON).
//! - **Opt-in gate**: Telemetry is disabled by default with an explicit toggle
//!   that takes effect immediately without restart.
//!
//! # Privacy Design
//!
//! All telemetry data is strictly hardware/benchmark data:
//! - No username, hostname, home directory paths
//! - No IP addresses, email addresses, or credentials
//! - No free-form user input
//!
//! All payload keys are from a predefined allow-list. The payload is validated
//! for PII before any submission attempt.

pub mod benchmark;
pub mod opt_in;
pub mod payload;
pub mod submit;

pub use benchmark::*;
pub use opt_in::*;
pub use payload::*;
pub use submit::*;
