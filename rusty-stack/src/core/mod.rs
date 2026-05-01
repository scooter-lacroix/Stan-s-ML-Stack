//! Core module — foundational types, contracts, and shared abstractions.
//!
//! This module provides the canonical type definitions used across all
//! other modules (platform, orchestrator, telemetry, adapter).

pub mod manifest;
pub mod plan;
pub mod telemetry_types;
pub mod types;
pub mod validation;
pub mod verification;

pub use manifest::*;
pub use plan::*;
pub use telemetry_types::*;
pub use types::*;
pub use validation::*;
pub use verification::*;
