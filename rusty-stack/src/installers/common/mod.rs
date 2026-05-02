//! Shared installer infrastructure modules.
//!
//! These modules provide common functionality used by all component installers:
//! - **package management**: unified interface for apt/dnf/pacman
//! - **distro detection**: thin facade over `platform::detection`
//! - **package mappings**: per-distro package name resolution
//! - **guard**: structured error handling, logging, progress, Python validation
//! - **utils**: color-aware output, command existence, print helpers
//! - **env_validation**: `.mlstack_env` checking with sensible defaults
//! - **rocm_env**: thin facade over `platform::linux` for ROCm paths/versions
//! - **ui_helper**: CLI arg parsing, path validation, git helpers
//!
//! # Validation Assertions
//!
//! - **VAL-INFRA-001**: Module structure compiles and integrates
//! - **VAL-INFRA-019**: No code duplication with platform modules

pub mod distro;
pub mod env_validation;
pub mod guard;
pub mod package_manager;
pub mod package_mappings;
pub mod rocm_env;
pub mod ui_helper;
pub mod utils;

// Re-export key types for convenience
pub use distro::DistroFacade;
pub use env_validation::{EnvValidationResult, EnvVars};
pub use guard::{
    detect_python_version, is_python_supported, is_python_supported_for_rocm_torch,
    log_error, log_info, log_warn, InstallerError, LogLevel, ProgressTracker, PythonVersion,
    MIN_PYTHON_VERSION, MAX_ROCM_TORCH_PYTHON,
};
pub use package_manager::{DryRunResult, PackageOperation, PackageManagerFacade};
pub use package_mappings::map_package_name;
pub use rocm_env::RocmEnv;
pub use ui_helper::{UiArgs, UiArgError, is_system_path};
pub use utils::{command_exists, get_colors, Colors, PythonPkgManager};
