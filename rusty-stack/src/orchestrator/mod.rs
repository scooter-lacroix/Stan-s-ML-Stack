//! Orchestrator module — upgrade and update command orchestration.
//!
//! This module provides the orchestration layer for CLI commands:
//! - [`upgrade`] — Application/runtime upgrade with version gating and rollback

pub mod upgrade;

pub use upgrade::*;
