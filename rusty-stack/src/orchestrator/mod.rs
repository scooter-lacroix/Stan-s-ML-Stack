//! Orchestrator module — upgrade and update command orchestration.
//!
//! This module provides the orchestration layer for CLI commands:
//! - [`upgrade`] — Application/runtime upgrade with version gating and rollback
//! - [`planner`] — Update plan computation, classification, and selection
//! - [`apply`] — Dependency-safe execution engine with failure isolation
//! - [`verify`] — Post-apply verification orchestration
//!
//! # Update Flow
//!
//! ```text
//! scan → manifest resolution → plan → user selection → apply → verify → report
//! ```
//!
//! The planner classifies updates as safe/guarded/blocked/candidate/experimental
//! based on semver, dependencies, hardware compatibility, and validation tier.
//! The apply engine executes in dependency-safe order, isolating failures so
//! unrelated work continues. The verify runner confirms each successful apply.

pub mod apply;
pub mod planner;
pub mod upgrade;
pub mod verify;

pub use apply::*;
pub use planner::*;
pub use upgrade::*;
pub use verify::*;
