//! Shared installer infrastructure modules.
//!
//! These modules provide common functionality used by all component installers:
//! package management, distro detection delegation, and package name mappings.
//!
//! # Validation Assertions
//!
//! - **VAL-INFRA-001**: Module structure compiles and integrates
//! - **VAL-INFRA-019**: No code duplication with platform modules

pub mod distro;
pub mod package_manager;
pub mod package_mappings;

// Re-export key types for convenience
pub use distro::DistroFacade;
pub use package_manager::{DryRunResult, PackageOperation, PackageManagerFacade};
pub use package_mappings::map_package_name;
