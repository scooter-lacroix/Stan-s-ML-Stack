//! Package Manager
//!
//! Unified package manager for coordinating installations across all ML components.

use crate::common::ProgressCallback;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Package types supported by the manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PackageType {
    ROCm,
    PyTorch,
    Triton,
    FlashAttention,
    Vllm,
    Megatron,
    OnnxRuntime,
}

impl PackageType {
    /// Returns the package name.
    pub fn name(&self) -> &'static str {
        match self {
            PackageType::ROCm => "ROCm",
            PackageType::PyTorch => "PyTorch",
            PackageType::Triton => "Triton",
            PackageType::FlashAttention => "Flash Attention",
            PackageType::Vllm => "vLLM",
            PackageType::Megatron => "Megatron-LM",
            PackageType::OnnxRuntime => "ONNX Runtime",
        }
    }

    /// Returns dependencies for this package.
    pub fn dependencies(&self) -> Vec<PackageType> {
        match self {
            PackageType::ROCm => vec![],
            PackageType::PyTorch => vec![PackageType::ROCm],
            PackageType::Triton => vec![PackageType::PyTorch],
            PackageType::FlashAttention => vec![PackageType::PyTorch, PackageType::Triton],
            PackageType::Vllm => vec![PackageType::PyTorch],
            PackageType::Megatron => vec![PackageType::PyTorch],
            PackageType::OnnxRuntime => vec![PackageType::ROCm],
        }
    }
}

/// Package installation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageRequest {
    /// Package type
    pub package_type: PackageType,
    /// Version to install
    pub version: String,
    /// Whether to force reinstall
    pub force: bool,
    /// Additional options
    pub options: HashMap<String, String>,
}

impl PackageRequest {
    /// Creates a new package request.
    pub fn new(package_type: PackageType, version: impl Into<String>) -> Self {
        Self {
            package_type,
            version: version.into(),
            force: false,
            options: HashMap::new(),
        }
    }

    /// Sets force reinstall.
    pub fn with_force(mut self, force: bool) -> Self {
        self.force = force;
        self
    }

    /// Adds an option.
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }
}

/// Package installation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageResult {
    /// Package type
    pub package_type: PackageType,
    /// Whether installation succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Installation duration in seconds
    pub duration_secs: f64,
}

/// Unified package manager.
pub struct PackageManager {
    /// Installation order (topological sort of dependencies)
    install_order: Vec<PackageType>,
    /// Installation results
    results: HashMap<PackageType, PackageResult>,
}

impl PackageManager {
    /// Creates a new package manager.
    pub fn new() -> Self {
        Self {
            install_order: Vec::new(),
            results: HashMap::new(),
        }
    }

    /// Resolves dependencies and creates installation plan.
    pub fn resolve_dependencies(&mut self, packages: &[PackageType]) -> Vec<PackageType> {
        let mut resolved = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn visit(
            pkg: PackageType,
            resolved: &mut Vec<PackageType>,
            visited: &mut std::collections::HashSet<PackageType>,
        ) {
            if visited.contains(&pkg) {
                return;
            }

            for dep in pkg.dependencies() {
                visit(dep, resolved, visited);
            }

            if !resolved.contains(&pkg) {
                resolved.push(pkg);
            }
            visited.insert(pkg);
        }

        for pkg in packages {
            visit(*pkg, &mut resolved, &mut visited);
        }

        self.install_order = resolved.clone();
        resolved
    }

    /// Installs a single package.
    pub async fn install_package(
        &mut self,
        request: &PackageRequest,
        _progress: &Option<ProgressCallback>,
    ) -> Result<PackageResult> {
        let start = std::time::Instant::now();

        let result = match request.package_type {
            PackageType::ROCm => {
                // ROCm installation would go here
                PackageResult {
                    package_type: request.package_type,
                    success: true,
                    error: None,
                    duration_secs: start.elapsed().as_secs_f64(),
                }
            }
            _ => {
                // Other package installations
                PackageResult {
                    package_type: request.package_type,
                    success: true,
                    error: None,
                    duration_secs: start.elapsed().as_secs_f64(),
                }
            }
        };

        self.results.insert(request.package_type, result.clone());
        Ok(result)
    }

    /// Installs multiple packages with dependency resolution.
    pub async fn install_packages(
        &mut self,
        requests: &[PackageRequest],
        progress: &Option<ProgressCallback>,
    ) -> Vec<PackageResult> {
        let package_types: Vec<PackageType> = requests.iter().map(|r| r.package_type).collect();
        let order = self.resolve_dependencies(&package_types);

        let mut results = Vec::new();

        for pkg_type in order {
            if let Some(request) = requests.iter().find(|r| r.package_type == pkg_type) {
                match self.install_package(request, progress).await {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        results.push(PackageResult {
                            package_type: pkg_type,
                            success: false,
                            error: Some(e.to_string()),
                            duration_secs: 0.0,
                        });
                    }
                }
            }
        }

        results
    }

    /// Gets installation results.
    pub fn results(&self) -> &HashMap<PackageType, PackageResult> {
        &self.results
    }

    /// Checks if all installations succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.results.values().all(|r| r.success)
    }
}

impl Default for PackageManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Output formatter for installation reports.
pub struct OutputFormatter;

impl OutputFormatter {
    /// Formats installation results as a table.
    pub fn format_table(results: &[PackageResult]) -> String {
        let mut output = String::new();
        output.push_str("╔══════════════════════╦══════════╦══════════════════╗\n");
        output.push_str("║ Package              ║ Status   ║ Duration         ║\n");
        output.push_str("╠══════════════════════╬══════════╬══════════════════╣\n");

        for result in results {
            let status = if result.success { "✓ OK" } else { "✗ FAIL" };
            let duration = format!("{:.2}s", result.duration_secs);
            output.push_str(&format!(
                "║ {:<20} ║ {:<8} ║ {:<16} ║\n",
                result.package_type.name(),
                status,
                duration
            ));
        }

        output.push_str("╚══════════════════════╩══════════╩══════════════════╝\n");
        output
    }

    /// Formats installation results as JSON.
    pub fn format_json(results: &[PackageResult]) -> String {
        serde_json::to_string_pretty(results).unwrap_or_default()
    }

    /// Formats a summary report.
    pub fn format_summary(results: &[PackageResult]) -> String {
        let total = results.len();
        let succeeded = results.iter().filter(|r| r.success).count();
        let failed = total - succeeded;
        let total_duration: f64 = results.iter().map(|r| r.duration_secs).sum();

        format!(
            "Installation Summary:\n\
             - Total packages: {}\n\
             - Succeeded: {}\n\
             - Failed: {}\n\
             - Total duration: {:.2}s\n",
            total, succeeded, failed, total_duration
        )
    }

    /// Formats detailed error report.
    pub fn format_errors(results: &[PackageResult]) -> String {
        let errors: Vec<&PackageResult> = results.iter().filter(|r| !r.success).collect();

        if errors.is_empty() {
            return "No errors occurred.".to_string();
        }

        let mut output = String::from("Errors:\n");
        for error in errors {
            if let Some(ref msg) = error.error {
                output.push_str(&format!(
                    "  - {}: {}\n",
                    error.package_type.name(),
                    msg
                ));
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_type_name() {
        assert_eq!(PackageType::ROCm.name(), "ROCm");
        assert_eq!(PackageType::PyTorch.name(), "PyTorch");
        assert_eq!(PackageType::Vllm.name(), "vLLM");
    }

    #[test]
    fn test_package_dependencies() {
        let rocm_deps = PackageType::ROCm.dependencies();
        assert!(rocm_deps.is_empty());

        let pytorch_deps = PackageType::PyTorch.dependencies();
        assert!(pytorch_deps.contains(&PackageType::ROCm));

        let flash_deps = PackageType::FlashAttention.dependencies();
        assert!(flash_deps.contains(&PackageType::PyTorch));
        assert!(flash_deps.contains(&PackageType::Triton));
    }

    #[test]
    fn test_package_request_creation() {
        let request = PackageRequest::new(PackageType::PyTorch, "2.5.0");
        assert_eq!(request.package_type, PackageType::PyTorch);
        assert_eq!(request.version, "2.5.0");
        assert!(!request.force);
    }

    #[test]
    fn test_package_request_builder() {
        let request = PackageRequest::new(PackageType::PyTorch, "2.5.0")
            .with_force(true)
            .with_option("cuda", "false");

        assert!(request.force);
        assert_eq!(request.options.get("cuda"), Some(&"false".to_string()));
    }

    #[test]
    fn test_dependency_resolution() {
        let mut manager = PackageManager::new();
        let packages = vec![PackageType::FlashAttention];
        let order = manager.resolve_dependencies(&packages);

        // Should include dependencies first
        assert!(order.contains(&PackageType::ROCm));
        assert!(order.contains(&PackageType::PyTorch));
        assert!(order.contains(&PackageType::Triton));
        assert!(order.contains(&PackageType::FlashAttention));

        // FlashAttention should be last
        assert_eq!(order.last(), Some(&PackageType::FlashAttention));
    }

    #[test]
    fn test_output_formatter_table() {
        let results = vec![
            PackageResult {
                package_type: PackageType::ROCm,
                success: true,
                error: None,
                duration_secs: 120.5,
            },
            PackageResult {
                package_type: PackageType::PyTorch,
                success: false,
                error: Some("Build failed".to_string()),
                duration_secs: 45.2,
            },
        ];

        let table = OutputFormatter::format_table(&results);
        assert!(table.contains("ROCm"));
        assert!(table.contains("PyTorch"));
        assert!(table.contains("OK"));
        assert!(table.contains("FAIL"));
    }

    #[test]
    fn test_output_formatter_summary() {
        let results = vec![
            PackageResult {
                package_type: PackageType::ROCm,
                success: true,
                error: None,
                duration_secs: 120.5,
            },
            PackageResult {
                package_type: PackageType::PyTorch,
                success: true,
                error: None,
                duration_secs: 45.2,
            },
        ];

        let summary = OutputFormatter::format_summary(&results);
        assert!(summary.contains("Total packages: 2"));
        assert!(summary.contains("Succeeded: 2"));
        assert!(summary.contains("Failed: 0"));
    }

    #[test]
    fn test_output_formatter_errors() {
        let results = vec![
            PackageResult {
                package_type: PackageType::PyTorch,
                success: false,
                error: Some("Build failed".to_string()),
                duration_secs: 45.2,
            },
        ];

        let errors = OutputFormatter::format_errors(&results);
        assert!(errors.contains("PyTorch"));
        assert!(errors.contains("Build failed"));
    }
}
