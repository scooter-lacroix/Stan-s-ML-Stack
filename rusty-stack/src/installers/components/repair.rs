//! Repair ML Stack utility — ports `scripts/repair_ml_stack.sh`.
//!
//! Calls correct sub-installers in sequence via native Rust modules,
//! propagates failures, and reports a summary of repair results.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-026**: Repair invokes correct sub-installers in order
//! - **VAL-INSTALL-027**: Repair propagates individual failures
//! - **VAL-INSTALL-051**: Repair dispatches to native Rust modules, not bash scripts

// ===========================================================================
// Types
// ===========================================================================

/// A constructed shell command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommand {
    /// The program to run.
    pub program: String,
    /// Arguments to pass.
    pub args: Vec<String>,
    /// Environment variables to set.
    pub env: Vec<(String, String)>,
}

impl ShellCommand {
    /// Format as a shell command string.
    pub fn to_command_string(&self) -> String {
        let env_prefix = self
            .env
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        let cmd = if self.args.is_empty() {
            self.program.clone()
        } else {
            format!("{} {}", self.program, self.args.join(" "))
        };
        if env_prefix.is_empty() {
            cmd
        } else {
            format!("{env_prefix} {cmd}")
        }
    }
}

/// Result of a single repair step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairStepResult {
    /// Name of the repair step.
    pub name: String,
    /// Whether the step succeeded.
    pub success: bool,
    /// Optional error message.
    pub error_message: Option<String>,
}

/// Result of the full repair sequence.
#[derive(Debug, Clone)]
pub struct RepairResult {
    /// Individual step results in order.
    pub steps: Vec<RepairStepResult>,
}

impl RepairResult {
    /// Create a new empty repair result.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step result.
    pub fn add_step(&mut self, name: &str, success: bool, error_message: Option<String>) {
        self.steps.push(RepairStepResult {
            name: name.to_string(),
            success,
            error_message,
        });
    }

    /// Get the number of successful steps.
    pub fn successful_count(&self) -> usize {
        self.steps.iter().filter(|s| s.success).count()
    }

    /// Get the number of failed steps.
    pub fn failed_count(&self) -> usize {
        self.steps.iter().filter(|s| !s.success).count()
    }

    /// Whether all steps succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.steps.iter().all(|s| s.success)
    }

    /// Get failed step names.
    pub fn failed_steps(&self) -> Vec<&str> {
        self.steps
            .iter()
            .filter(|s| !s.success)
            .map(|s| s.name.as_str())
            .collect()
    }

    /// Format a summary report.
    pub fn format_summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("ML Stack Repair Summary".to_string());
        lines.push("=======================".to_string());
        for step in &self.steps {
            let status = if step.success { "✓" } else { "✗" };
            let msg = match &step.error_message {
                Some(m) => format!(" ({})", m),
                None => String::new(),
            };
            lines.push(format!("{} {}{}", status, step.name, msg));
        }
        lines.push(String::new());
        lines.push(format!(
            "Total: {} succeeded, {} failed",
            self.successful_count(),
            self.failed_count()
        ));
        lines.join("\n")
    }
}

impl Default for RepairResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Repair step identifier — matches the sequence in repair_ml_stack.sh.
///
/// The original script's main() repair sequence is:
/// 1. ROCm installation (fix_rocm_installation)
/// 2. AMDGPU drivers (fix_amdgpu_drivers)
/// 3. PyTorch (fix_pytorch_installation)
/// 4. ML Stack Core (fix_ml_stack_core)
/// 5. AITER (fix_aiter_installation)
/// 6. MIGraphX Python (fix_migraphx_python_installation)
/// 7. DeepSpeed (fix_deepspeed_installation)
/// 8. Environment variables (fix_environment_variables)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepairStep {
    /// Fix ROCm installation.
    Rocm,
    /// Fix AMDGPU drivers.
    AmdgpuDrivers,
    /// Fix PyTorch installation.
    PyTorch,
    /// Fix ML Stack Core components.
    MlStackCore,
    /// Fix AITER installation.
    Aiter,
    /// Fix MIGraphX Python bindings.
    MigraphxPython,
    /// Fix DeepSpeed installation.
    DeepSpeed,
    /// Fix environment variables.
    EnvironmentVariables,
}

impl RepairStep {
    /// Get all repair steps in the correct order.
    pub fn all_in_order() -> Vec<RepairStep> {
        vec![
            RepairStep::Rocm,
            RepairStep::AmdgpuDrivers,
            RepairStep::PyTorch,
            RepairStep::MlStackCore,
            RepairStep::Aiter,
            RepairStep::MigraphxPython,
            RepairStep::DeepSpeed,
            RepairStep::EnvironmentVariables,
        ]
    }

    /// Get the display name of the repair step.
    pub fn name(&self) -> &'static str {
        match self {
            RepairStep::Rocm => "ROCm Installation",
            RepairStep::AmdgpuDrivers => "AMDGPU Drivers",
            RepairStep::PyTorch => "PyTorch Installation",
            RepairStep::MlStackCore => "ML Stack Core",
            RepairStep::Aiter => "AITER Installation",
            RepairStep::MigraphxPython => "MIGraphX Python",
            RepairStep::DeepSpeed => "DeepSpeed Installation",
            RepairStep::EnvironmentVariables => "Environment Variables",
        }
    }

    /// Get the native component ID that this repair step dispatches to.
    /// Each repair step re-runs the corresponding native installer module.
    pub fn native_component_id(&self) -> &'static str {
        match self {
            RepairStep::Rocm => "rocm",
            RepairStep::AmdgpuDrivers => "amdgpu-drivers",
            RepairStep::PyTorch => "pytorch",
            RepairStep::MlStackCore => "ml-stack-core",
            RepairStep::Aiter => "aiter",
            RepairStep::MigraphxPython => "migraphx-python",
            RepairStep::DeepSpeed => "deepspeed",
            RepairStep::EnvironmentVariables => "permanent-env",
        }
    }

    /// Legacy script name (kept for documentation/test reference only).
    #[allow(dead_code)]
    pub fn legacy_script_name(&self) -> &'static str {
        match self {
            RepairStep::Rocm => "install_rocm.sh",
            RepairStep::AmdgpuDrivers => "install_amdgpu_drivers.sh",
            RepairStep::PyTorch => "install_pytorch_rocm.sh",
            RepairStep::MlStackCore => "install_ml_stack.sh",
            RepairStep::Aiter => "install_aiter.sh",
            RepairStep::MigraphxPython => "install_migraphx_python.sh",
            RepairStep::DeepSpeed => "install_deepspeed.sh",
            RepairStep::EnvironmentVariables => "setup_permanent_rocm_env.sh",
        }
    }
}

/// Configuration for the repair tool.
#[derive(Debug, Clone)]
pub struct RepairConfig {
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Whether to run in non-interactive mode.
    pub non_interactive: bool,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            python_bin: "python3".to_string(),
            dry_run: false,
            non_interactive: false,
        }
    }
}

/// The ML Stack repair tool.
pub struct RepairInstaller {
    #[allow(dead_code)]
    config: RepairConfig,
}

impl RepairInstaller {
    /// Create a new repair tool with the given config.
    pub fn new(config: RepairConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(RepairConfig::default())
    }

    // -----------------------------------------------------------------------
    // Step enumeration (VAL-INSTALL-026)
    // -----------------------------------------------------------------------

    /// Get the repair step sequence in the correct order.
    pub fn repair_sequence() -> Vec<RepairStep> {
        RepairStep::all_in_order()
    }

    /// Get the native component IDs for all repair steps, in order.
    pub fn native_component_ids() -> Vec<&'static str> {
        Self::repair_sequence()
            .iter()
            .map(|s| s.native_component_id())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Failure propagation (VAL-INSTALL-027)
    // -----------------------------------------------------------------------

    /// Run the repair sequence with mock step results.
    /// Continues through all steps even if some fail, collecting results.
    /// Returns a summary with all step outcomes.
    pub fn run_repair_sequence<F>(&self, mut step_runner: F) -> RepairResult
    where
        F: FnMut(RepairStep) -> Result<(), String>,
    {
        let mut result = RepairResult::new();

        for step in Self::repair_sequence() {
            match step_runner(step) {
                Ok(()) => result.add_step(step.name(), true, None),
                Err(msg) => result.add_step(step.name(), false, Some(msg)),
            }
        }

        result
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repair_sequence_order_matches_original_script() {
        let sequence = RepairInstaller::repair_sequence();
        assert_eq!(sequence.len(), 8);
        assert_eq!(sequence[0], RepairStep::Rocm);
        assert_eq!(sequence[1], RepairStep::AmdgpuDrivers);
        assert_eq!(sequence[2], RepairStep::PyTorch);
        assert_eq!(sequence[3], RepairStep::MlStackCore);
        assert_eq!(sequence[4], RepairStep::Aiter);
        assert_eq!(sequence[5], RepairStep::MigraphxPython);
        assert_eq!(sequence[6], RepairStep::DeepSpeed);
        assert_eq!(sequence[7], RepairStep::EnvironmentVariables);
    }

    #[test]
    fn test_repair_step_names() {
        assert_eq!(RepairStep::Rocm.name(), "ROCm Installation");
        assert_eq!(RepairStep::AmdgpuDrivers.name(), "AMDGPU Drivers");
        assert_eq!(RepairStep::PyTorch.name(), "PyTorch Installation");
        assert_eq!(RepairStep::MlStackCore.name(), "ML Stack Core");
        assert_eq!(RepairStep::Aiter.name(), "AITER Installation");
        assert_eq!(RepairStep::MigraphxPython.name(), "MIGraphX Python");
        assert_eq!(RepairStep::DeepSpeed.name(), "DeepSpeed Installation");
        assert_eq!(
            RepairStep::EnvironmentVariables.name(),
            "Environment Variables"
        );
    }

    #[test]
    fn test_repair_step_native_component_ids() {
        assert_eq!(RepairStep::Rocm.native_component_id(), "rocm");
        assert_eq!(
            RepairStep::AmdgpuDrivers.native_component_id(),
            "amdgpu-drivers"
        );
        assert_eq!(RepairStep::PyTorch.native_component_id(), "pytorch");
        assert_eq!(
            RepairStep::MlStackCore.native_component_id(),
            "ml-stack-core"
        );
        assert_eq!(RepairStep::Aiter.native_component_id(), "aiter");
        assert_eq!(
            RepairStep::MigraphxPython.native_component_id(),
            "migraphx-python"
        );
        assert_eq!(RepairStep::DeepSpeed.native_component_id(), "deepspeed");
        assert_eq!(
            RepairStep::EnvironmentVariables.native_component_id(),
            "permanent-env"
        );
    }

    #[test]
    fn test_native_component_ids_match_repair_sequence() {
        let ids = RepairInstaller::native_component_ids();
        let steps = RepairInstaller::repair_sequence();
        assert_eq!(ids.len(), steps.len());
        for (step, id) in steps.iter().zip(ids.iter()) {
            assert_eq!(step.native_component_id(), *id);
        }
    }

    #[test]
    fn test_repair_uses_native_dispatch_not_scripts() {
        // Verify every repair step maps to a native component ID
        // (no bash script references needed for repair)
        for step in RepairInstaller::repair_sequence() {
            let id = step.native_component_id();
            assert!(
                crate::installers::components::is_native_component(id),
                "Repair step '{}' maps to component '{}' which must be native",
                step.name(),
                id
            );
        }
    }

    #[test]
    fn test_repair_propagates_failures() {
        let repair = RepairInstaller::with_defaults();
        let result = repair.run_repair_sequence(|step| match step {
            RepairStep::PyTorch => Err("PyTorch installation failed".to_string()),
            RepairStep::DeepSpeed => Err("DeepSpeed build error".to_string()),
            _ => Ok(()),
        });

        assert_eq!(result.failed_count(), 2);
        assert_eq!(result.successful_count(), 6);
        assert!(!result.all_succeeded());

        let failed = result.failed_steps();
        assert!(failed.contains(&"PyTorch Installation"));
        assert!(failed.contains(&"DeepSpeed Installation"));
    }

    #[test]
    fn test_repair_all_succeed() {
        let repair = RepairInstaller::with_defaults();
        let result = repair.run_repair_sequence(|_| Ok(()));

        assert!(result.all_succeeded());
        assert_eq!(result.successful_count(), 8);
        assert_eq!(result.failed_count(), 0);
    }

    #[test]
    fn test_repair_summary_format() {
        let repair = RepairInstaller::with_defaults();
        let result = repair.run_repair_sequence(|step| match step {
            RepairStep::PyTorch => Err("timeout".to_string()),
            _ => Ok(()),
        });

        let summary = result.format_summary();
        assert!(summary.contains("ML Stack Repair Summary"));
        assert!(summary.contains("7 succeeded, 1 failed"));
        assert!(summary.contains("✗ PyTorch Installation (timeout)"));
        assert!(summary.contains("✓ ROCm Installation"));
    }

    #[test]
    fn test_repair_continues_after_failure() {
        let repair = RepairInstaller::with_defaults();
        let mut visited = Vec::new();

        let _result = repair.run_repair_sequence(|step| {
            visited.push(step);
            if step == RepairStep::Rocm {
                Err("ROCm failed".to_string())
            } else {
                Ok(())
            }
        });

        assert_eq!(visited.len(), 8);
        assert_eq!(visited[0], RepairStep::Rocm);
        assert_eq!(visited[7], RepairStep::EnvironmentVariables);
    }

    #[test]
    fn test_repair_result_default() {
        let result = RepairResult::default();
        assert_eq!(result.steps.len(), 0);
        assert!(result.all_succeeded());
    }
}
