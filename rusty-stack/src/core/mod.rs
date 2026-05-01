//! Core module — foundational types, contracts, and shared abstractions.
//!
//! This module provides the canonical type definitions used across all
//! other modules (platform, orchestrator, telemetry, adapter).

pub mod manifest;
pub mod types;

pub use manifest::*;
pub use types::*;
