//! Orchestrator module — upgrade and update command orchestration.
//!
//! This module provides the orchestration layer for CLI commands:
//! - [`upgrade`] — Application/runtime upgrade with version gating and rollback
//! - [`planner`] — Update plan computation, classification, and selection
//!
//! # Update Flow
//!
//! ```text
//! scan → manifest resolution → plan → user selection → apply → verify → report
//! ```
//!
//! The planner classifies updates as safe/guarded/blocked/candidate/experimental
//! based on semver, dependencies, hardware compatibility, and validation tier.

pub mod planner;
pub mod upgrade;

pub use planner::*;
pub use upgrade::*;
