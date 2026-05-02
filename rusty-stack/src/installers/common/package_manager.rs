//! Package manager abstraction for install/remove/query operations.
//!
//! Provides a unified interface for system package managers (apt, dnf, pacman,
//! yum, zypper) with dry-run support. Command construction is tested without
//! actually executing subprocesses.
//!
//! # Validation Assertions
//!
//! - **VAL-INFRA-005**: Package manager operations correct
//! - **VAL-INFRA-006**: Package manager supports dry-run mode

use crate::platform::detection::PackageManager;
use std::fmt;
use std::process::Command;

/// Type of package operation being performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackageOperation {
    Install,
    Remove,
    Purge,
    Update,
    Upgrade,
    Search,
    Info,
    IsInstalled,
    ListInstalled,
}

impl fmt::Display for PackageOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackageOperation::Install => write!(f, "install"),
            PackageOperation::Remove => write!(f, "remove"),
            PackageOperation::Purge => write!(f, "purge"),
            PackageOperation::Update => write!(f, "update"),
            PackageOperation::Upgrade => write!(f, "upgrade"),
            PackageOperation::Search => write!(f, "search"),
            PackageOperation::Info => write!(f, "info"),
            PackageOperation::IsInstalled => write!(f, "is_installed"),
            PackageOperation::ListInstalled => write!(f, "list_installed"),
        }
    }
}

/// Result of a dry-run operation — contains the command that would be executed.
#[derive(Debug, Clone)]
pub struct DryRunResult {
    /// The program that would be executed (e.g., "sudo", "apt").
    pub program: String,
    /// The arguments that would be passed.
    pub args: Vec<String>,
    /// Human-readable description of the operation.
    pub description: String,
}

impl DryRunResult {
    /// Format the full command as a string.
    pub fn to_command_string(&self) -> String {
        if self.args.is_empty() {
            self.program.clone()
        } else {
            format!("{} {}", self.program, self.args.join(" "))
        }
    }
}

impl fmt::Display for DryRunResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[DRY-RUN] {}", self.to_command_string())
    }
}

/// Facade providing package manager operations with optional dry-run mode.
#[derive(Debug, Clone)]
pub struct PackageManagerFacade {
    /// The package manager to use.
    pkg_manager: PackageManager,
    /// Whether to run in dry-run mode (log commands without executing).
    dry_run: bool,
    /// Whether to translate package names via mappings.
    translate_packages: bool,
}

impl PackageManagerFacade {
    /// Create a new facade for the given package manager.
    pub fn new(pkg_manager: PackageManager) -> Self {
        Self {
            pkg_manager,
            dry_run: false,
            translate_packages: true,
        }
    }

    /// Enable or disable dry-run mode.
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Enable or disable package name translation.
    pub fn with_translate_packages(mut self, translate: bool) -> Self {
        self.translate_packages = translate;
        self
    }

    /// Get the package manager type.
    pub fn package_manager(&self) -> PackageManager {
        self.pkg_manager
    }

    /// Check if dry-run mode is enabled.
    pub fn is_dry_run(&self) -> bool {
        self.dry_run
    }

    // =====================================================================
    // Command Construction (public, testable without subprocesses)
    // =====================================================================

    /// Build the command for updating package indices.
    ///
    /// Returns the program and args (without sudo prefix).
    pub fn build_update_indices_cmd(&self) -> (String, Vec<String>) {
        match self.pkg_manager {
            PackageManager::Apt => (
                "apt".to_string(),
                vec!["update".to_string(), "-qq".to_string()],
            ),
            PackageManager::Pacman => (
                "pacman".to_string(),
                vec!["-Sy".to_string(), "--noconfirm".to_string()],
            ),
            PackageManager::Dnf => ("dnf".to_string(), vec!["makecache".to_string()]),
            PackageManager::Yum => ("yum".to_string(), vec!["makecache".to_string()]),
            PackageManager::Zypper => ("zypper".to_string(), vec!["refresh".to_string()]),
            PackageManager::Unknown => (
                "echo".to_string(),
                vec!["unknown package manager".to_string()],
            ),
        }
    }

    /// Build the command for upgrading all packages.
    pub fn build_upgrade_cmd(&self) -> (String, Vec<String>) {
        match self.pkg_manager {
            PackageManager::Apt => (
                "apt".to_string(),
                vec!["upgrade".to_string(), "-y".to_string(), "-qq".to_string()],
            ),
            PackageManager::Pacman => (
                "pacman".to_string(),
                vec!["-Su".to_string(), "--noconfirm".to_string()],
            ),
            PackageManager::Dnf => (
                "dnf".to_string(),
                vec![
                    "upgrade".to_string(),
                    "-y".to_string(),
                    "--refresh".to_string(),
                ],
            ),
            PackageManager::Yum => (
                "yum".to_string(),
                vec!["upgrade".to_string(), "-y".to_string()],
            ),
            PackageManager::Zypper => (
                "zypper".to_string(),
                vec!["update".to_string(), "-y".to_string()],
            ),
            PackageManager::Unknown => (
                "echo".to_string(),
                vec!["unknown package manager".to_string()],
            ),
        }
    }

    /// Build the command for installing packages.
    ///
    /// Package names are translated if translation is enabled.
    pub fn build_install_cmd(&self, packages: &[&str]) -> (String, Vec<String>) {
        let native = self.translate_package_names(packages);
        match self.pkg_manager {
            PackageManager::Apt => (
                "apt".to_string(),
                vec!["install".to_string(), "-y".to_string(), "-qq".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Pacman => (
                "pacman".to_string(),
                vec![
                    "-S".to_string(),
                    "--noconfirm".to_string(),
                    "--needed".to_string(),
                ]
                .into_iter()
                .chain(native)
                .collect(),
            ),
            PackageManager::Dnf => (
                "dnf".to_string(),
                vec!["install".to_string(), "-y".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Yum => (
                "yum".to_string(),
                vec!["install".to_string(), "-y".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Zypper => (
                "zypper".to_string(),
                vec!["install".to_string(), "-y".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Unknown => (
                "echo".to_string(),
                vec!["unknown package manager".to_string()],
            ),
        }
    }

    /// Build the command for removing packages.
    pub fn build_remove_cmd(&self, packages: &[&str]) -> (String, Vec<String>) {
        let native = self.translate_package_names(packages);
        match self.pkg_manager {
            PackageManager::Apt => (
                "apt".to_string(),
                vec!["remove".to_string(), "-y".to_string(), "-qq".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Pacman => (
                "pacman".to_string(),
                vec!["-R".to_string(), "--noconfirm".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Dnf => (
                "dnf".to_string(),
                vec!["remove".to_string(), "-y".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Yum => (
                "yum".to_string(),
                vec!["remove".to_string(), "-y".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Zypper => (
                "zypper".to_string(),
                vec!["remove".to_string(), "-y".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Unknown => (
                "echo".to_string(),
                vec!["unknown package manager".to_string()],
            ),
        }
    }

    /// Build the command for purging packages (remove with config files).
    pub fn build_purge_cmd(&self, packages: &[&str]) -> (String, Vec<String>) {
        let native = self.translate_package_names(packages);
        match self.pkg_manager {
            PackageManager::Apt => (
                "apt".to_string(),
                vec!["purge".to_string(), "-y".to_string(), "-qq".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Pacman => (
                "pacman".to_string(),
                vec!["-Rn".to_string(), "--noconfirm".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            // dnf/yum/zypper don't distinguish between remove and purge
            PackageManager::Dnf => (
                "dnf".to_string(),
                vec!["remove".to_string(), "-y".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Yum => (
                "yum".to_string(),
                vec!["remove".to_string(), "-y".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Zypper => (
                "zypper".to_string(),
                vec!["remove".to_string(), "-y".to_string()]
                    .into_iter()
                    .chain(native)
                    .collect(),
            ),
            PackageManager::Unknown => (
                "echo".to_string(),
                vec!["unknown package manager".to_string()],
            ),
        }
    }

    /// Build the command for checking if a package is installed.
    ///
    /// Returns the program and args for the query command.
    /// The package name is translated if translation is enabled.
    pub fn build_is_installed_cmd(&self, package: &str) -> (String, Vec<String>) {
        let native = self.translate_single(package);
        match self.pkg_manager {
            PackageManager::Apt => ("dpkg".to_string(), vec!["-l".to_string(), native]),
            PackageManager::Pacman => ("pacman".to_string(), vec!["-Qi".to_string(), native]),
            PackageManager::Dnf | PackageManager::Yum | PackageManager::Zypper => {
                ("rpm".to_string(), vec!["-q".to_string(), native])
            }
            PackageManager::Unknown => (
                "echo".to_string(),
                vec!["unknown package manager".to_string()],
            ),
        }
    }

    /// Build the command for searching packages.
    pub fn build_search_cmd(&self, query: &str) -> (String, Vec<String>) {
        match self.pkg_manager {
            PackageManager::Apt => (
                "apt".to_string(),
                vec!["search".to_string(), query.to_string()],
            ),
            PackageManager::Pacman => (
                "pacman".to_string(),
                vec!["-Ss".to_string(), query.to_string()],
            ),
            PackageManager::Dnf => (
                "dnf".to_string(),
                vec!["search".to_string(), query.to_string()],
            ),
            PackageManager::Yum => (
                "yum".to_string(),
                vec!["search".to_string(), query.to_string()],
            ),
            PackageManager::Zypper => (
                "zypper".to_string(),
                vec!["search".to_string(), query.to_string()],
            ),
            PackageManager::Unknown => (
                "echo".to_string(),
                vec!["unknown package manager".to_string()],
            ),
        }
    }

    /// Build the command for showing package info.
    pub fn build_info_cmd(&self, package: &str) -> (String, Vec<String>) {
        let native = self.translate_single(package);
        match self.pkg_manager {
            PackageManager::Apt => ("apt".to_string(), vec!["show".to_string(), native]),
            PackageManager::Pacman => ("pacman".to_string(), vec!["-Si".to_string(), native]),
            PackageManager::Dnf => ("dnf".to_string(), vec!["info".to_string(), native]),
            PackageManager::Yum => ("yum".to_string(), vec!["info".to_string(), native]),
            PackageManager::Zypper => ("zypper".to_string(), vec!["info".to_string(), native]),
            PackageManager::Unknown => (
                "echo".to_string(),
                vec!["unknown package manager".to_string()],
            ),
        }
    }

    // =====================================================================
    // Execution Methods
    // =====================================================================

    /// Update package indices.
    ///
    /// In dry-run mode, returns a `DryRunResult` without executing.
    pub fn update_indices(&self) -> Result<Option<DryRunResult>, anyhow::Error> {
        let (program, args) = self.build_update_indices_cmd();
        self.execute_with_sudo(&program, &args, "update package indices")
    }

    /// Install packages.
    ///
    /// In dry-run mode, returns a `DryRunResult` without executing.
    pub fn install(&self, packages: &[&str]) -> Result<Option<DryRunResult>, anyhow::Error> {
        if packages.is_empty() {
            anyhow::bail!("No packages specified for install");
        }
        let (program, args) = self.build_install_cmd(packages);
        let desc = format!("install packages: {}", packages.join(", "));
        self.execute_with_sudo(&program, &args, &desc)
    }

    /// Remove packages.
    pub fn remove(&self, packages: &[&str]) -> Result<Option<DryRunResult>, anyhow::Error> {
        if packages.is_empty() {
            anyhow::bail!("No packages specified for remove");
        }
        let (program, args) = self.build_remove_cmd(packages);
        let desc = format!("remove packages: {}", packages.join(", "));
        self.execute_with_sudo(&program, &args, &desc)
    }

    /// Purge packages (remove with configuration files).
    pub fn purge(&self, packages: &[&str]) -> Result<Option<DryRunResult>, anyhow::Error> {
        if packages.is_empty() {
            anyhow::bail!("No packages specified for purge");
        }
        let (program, args) = self.build_purge_cmd(packages);
        let desc = format!("purge packages: {}", packages.join(", "));
        self.execute_with_sudo(&program, &args, &desc)
    }

    /// Check if a package is installed.
    ///
    /// Returns `Ok(true)` if installed, `Ok(false)` if not.
    /// In dry-run mode, returns `Ok(false)` with the dry-run result.
    pub fn is_installed(&self, package: &str) -> Result<bool, anyhow::Error> {
        let (program, args) = self.build_is_installed_cmd(package);

        if self.dry_run {
            eprintln!("[DRY-RUN] {} {}", program, args.join(" "));
            return Ok(false);
        }

        let status = Command::new(&program).args(&args).output()?;
        Ok(status.status.success())
    }

    /// Search for packages.
    pub fn search(&self, query: &str) -> Result<Option<String>, anyhow::Error> {
        let (program, args) = self.build_search_cmd(query);

        if self.dry_run {
            eprintln!("[DRY-RUN] {} {}", program, args.join(" "));
            return Ok(None);
        }

        let output = Command::new(&program).args(&args).output()?;
        Ok(Some(String::from_utf8_lossy(&output.stdout).to_string()))
    }

    /// Get package info.
    pub fn info(&self, package: &str) -> Result<Option<String>, anyhow::Error> {
        let (program, args) = self.build_info_cmd(package);

        if self.dry_run {
            eprintln!("[DRY-RUN] {} {}", program, args.join(" "));
            return Ok(None);
        }

        let output = Command::new(&program).args(&args).output()?;
        Ok(Some(String::from_utf8_lossy(&output.stdout).to_string()))
    }

    // =====================================================================
    // Internal Helpers
    // =====================================================================

    /// Translate package names if translation is enabled.
    fn translate_package_names(&self, packages: &[&str]) -> Vec<String> {
        if self.translate_packages {
            super::package_mappings::map_package_names(packages, self.pkg_manager)
        } else {
            packages.iter().map(|s| s.to_string()).collect()
        }
    }

    /// Translate a single package name if translation is enabled.
    fn translate_single(&self, package: &str) -> String {
        if self.translate_packages {
            super::package_mappings::map_package_name(package, self.pkg_manager)
        } else {
            package.to_string()
        }
    }

    /// Execute a command with sudo, or return a dry-run result.
    fn execute_with_sudo(
        &self,
        program: &str,
        args: &[String],
        description: &str,
    ) -> Result<Option<DryRunResult>, anyhow::Error> {
        if self.dry_run {
            let result = DryRunResult {
                program: format!("sudo {}", program),
                args: args.to_vec(),
                description: description.to_string(),
            };
            eprintln!("{}", result);
            return Ok(Some(result));
        }

        let status = Command::new("sudo")
            .arg(program)
            .args(args)
            .env("DEBIAN_FRONTEND", "noninteractive")
            .status()?;

        if !status.success() {
            anyhow::bail!("Failed to {}: exit code {:?}", description, status.code());
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =====================================================================
    // VAL-INFRA-005: Package manager operations correct
    // =====================================================================

    // --- Update Indices ---

    #[test]
    fn test_update_indices_apt() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let (prog, args) = pm.build_update_indices_cmd();
        assert_eq!(prog, "apt");
        assert!(args.contains(&"update".to_string()));
    }

    #[test]
    fn test_update_indices_pacman() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman);
        let (prog, args) = pm.build_update_indices_cmd();
        assert_eq!(prog, "pacman");
        assert!(args.contains(&"-Sy".to_string()));
        assert!(args.contains(&"--noconfirm".to_string()));
    }

    #[test]
    fn test_update_indices_dnf() {
        let pm = PackageManagerFacade::new(PackageManager::Dnf);
        let (prog, args) = pm.build_update_indices_cmd();
        assert_eq!(prog, "dnf");
        assert!(args.contains(&"makecache".to_string()));
    }

    #[test]
    fn test_update_indices_yum() {
        let pm = PackageManagerFacade::new(PackageManager::Yum);
        let (prog, args) = pm.build_update_indices_cmd();
        assert_eq!(prog, "yum");
        assert!(args.contains(&"makecache".to_string()));
    }

    #[test]
    fn test_update_indices_zypper() {
        let pm = PackageManagerFacade::new(PackageManager::Zypper);
        let (prog, args) = pm.build_update_indices_cmd();
        assert_eq!(prog, "zypper");
        assert!(args.contains(&"refresh".to_string()));
    }

    // --- Install ---

    #[test]
    fn test_install_apt() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let (prog, args) = pm.build_install_cmd(&["build-essential", "cmake"]);
        assert_eq!(prog, "apt");
        assert!(args.contains(&"install".to_string()));
        assert!(args.contains(&"-y".to_string()));
        assert!(args.contains(&"build-essential".to_string()));
        assert!(args.contains(&"cmake".to_string()));
    }

    #[test]
    fn test_install_pacman_with_translation() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman);
        let (prog, args) = pm.build_install_cmd(&["build-essential"]);
        assert_eq!(prog, "pacman");
        assert!(args.contains(&"-S".to_string()));
        assert!(args.contains(&"--noconfirm".to_string()));
        assert!(args.contains(&"--needed".to_string()));
        // build-essential → base-devel on pacman
        assert!(args.contains(&"base-devel".to_string()));
    }

    #[test]
    fn test_install_dnf() {
        let pm = PackageManagerFacade::new(PackageManager::Dnf);
        let (prog, args) = pm.build_install_cmd(&["cmake"]);
        assert_eq!(prog, "dnf");
        assert!(args.contains(&"install".to_string()));
        assert!(args.contains(&"-y".to_string()));
        assert!(args.contains(&"cmake".to_string()));
    }

    #[test]
    fn test_install_yum() {
        let pm = PackageManagerFacade::new(PackageManager::Yum);
        let (prog, args) = pm.build_install_cmd(&["git"]);
        assert_eq!(prog, "yum");
        assert!(args.contains(&"install".to_string()));
        assert!(args.contains(&"-y".to_string()));
    }

    #[test]
    fn test_install_zypper() {
        let pm = PackageManagerFacade::new(PackageManager::Zypper);
        let (prog, args) = pm.build_install_cmd(&["git"]);
        assert_eq!(prog, "zypper");
        assert!(args.contains(&"install".to_string()));
        assert!(args.contains(&"-y".to_string()));
    }

    // --- Remove ---

    #[test]
    fn test_remove_apt() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let (prog, args) = pm.build_remove_cmd(&["nano"]);
        assert_eq!(prog, "apt");
        assert!(args.contains(&"remove".to_string()));
        assert!(args.contains(&"-y".to_string()));
        assert!(args.contains(&"nano".to_string()));
    }

    #[test]
    fn test_remove_pacman() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman);
        let (prog, args) = pm.build_remove_cmd(&["nano"]);
        assert_eq!(prog, "pacman");
        assert!(args.contains(&"-R".to_string()));
        assert!(args.contains(&"--noconfirm".to_string()));
    }

    #[test]
    fn test_remove_dnf() {
        let pm = PackageManagerFacade::new(PackageManager::Dnf);
        let (prog, args) = pm.build_remove_cmd(&["nano"]);
        assert_eq!(prog, "dnf");
        assert!(args.contains(&"remove".to_string()));
    }

    // --- Purge ---

    #[test]
    fn test_purge_apt() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let (prog, args) = pm.build_purge_cmd(&["nginx"]);
        assert_eq!(prog, "apt");
        assert!(args.contains(&"purge".to_string()));
    }

    #[test]
    fn test_purge_pacman() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman);
        let (prog, args) = pm.build_purge_cmd(&["nginx"]);
        assert_eq!(prog, "pacman");
        assert!(args.contains(&"-Rn".to_string()));
    }

    #[test]
    fn test_purge_dnf_uses_remove() {
        let pm = PackageManagerFacade::new(PackageManager::Dnf);
        let (prog, args) = pm.build_purge_cmd(&["nginx"]);
        assert_eq!(prog, "dnf");
        assert!(args.contains(&"remove".to_string()));
    }

    // --- Is Installed ---

    #[test]
    fn test_is_installed_apt() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let (prog, args) = pm.build_is_installed_cmd("curl");
        assert_eq!(prog, "dpkg");
        assert!(args.contains(&"-l".to_string()));
        assert!(args.contains(&"curl".to_string()));
    }

    #[test]
    fn test_is_installed_pacman() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman);
        let (prog, args) = pm.build_is_installed_cmd("curl");
        assert_eq!(prog, "pacman");
        assert!(args.contains(&"-Qi".to_string()));
        assert!(args.contains(&"curl".to_string()));
    }

    #[test]
    fn test_is_installed_dnf() {
        let pm = PackageManagerFacade::new(PackageManager::Dnf);
        let (prog, args) = pm.build_is_installed_cmd("curl");
        assert_eq!(prog, "rpm");
        assert!(args.contains(&"-q".to_string()));
    }

    #[test]
    fn test_is_installed_yum() {
        let pm = PackageManagerFacade::new(PackageManager::Yum);
        let (prog, _args) = pm.build_is_installed_cmd("curl");
        assert_eq!(prog, "rpm");
    }

    #[test]
    fn test_is_installed_zypper() {
        let pm = PackageManagerFacade::new(PackageManager::Zypper);
        let (prog, _args) = pm.build_is_installed_cmd("curl");
        assert_eq!(prog, "rpm");
    }

    // --- Search ---

    #[test]
    fn test_search_apt() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let (prog, args) = pm.build_search_cmd("python");
        assert_eq!(prog, "apt");
        assert!(args.contains(&"search".to_string()));
        assert!(args.contains(&"python".to_string()));
    }

    #[test]
    fn test_search_pacman() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman);
        let (prog, args) = pm.build_search_cmd("python");
        assert_eq!(prog, "pacman");
        assert!(args.contains(&"-Ss".to_string()));
    }

    // --- Info ---

    #[test]
    fn test_info_apt() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let (prog, args) = pm.build_info_cmd("python3");
        assert_eq!(prog, "apt");
        assert!(args.contains(&"show".to_string()));
    }

    #[test]
    fn test_info_pacman() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman);
        let (prog, args) = pm.build_info_cmd("python3");
        assert_eq!(prog, "pacman");
        assert!(args.contains(&"-Si".to_string()));
    }

    #[test]
    fn test_info_dnf() {
        let pm = PackageManagerFacade::new(PackageManager::Dnf);
        let (prog, args) = pm.build_info_cmd("python3");
        assert_eq!(prog, "dnf");
        assert!(args.contains(&"info".to_string()));
    }

    // --- Upgrade ---

    #[test]
    fn test_upgrade_apt() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let (prog, args) = pm.build_upgrade_cmd();
        assert_eq!(prog, "apt");
        assert!(args.contains(&"upgrade".to_string()));
        assert!(args.contains(&"-y".to_string()));
    }

    #[test]
    fn test_upgrade_pacman() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman);
        let (prog, args) = pm.build_upgrade_cmd();
        assert_eq!(prog, "pacman");
        assert!(args.contains(&"-Su".to_string()));
    }

    // --- Translation in commands ---

    #[test]
    fn test_install_translates_package_names() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman);
        let (_, args) = pm.build_install_cmd(&["build-essential", "python3-dev"]);
        // build-essential → base-devel, python3-dev → python
        assert!(args.contains(&"base-devel".to_string()));
        assert!(args.contains(&"python".to_string()));
    }

    #[test]
    fn test_install_no_translation() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman).with_translate_packages(false);
        let (_, args) = pm.build_install_cmd(&["build-essential"]);
        // Should use original name
        assert!(args.contains(&"build-essential".to_string()));
        assert!(!args.contains(&"base-devel".to_string()));
    }

    // --- Unknown package manager ---

    #[test]
    fn test_unknown_pm_install() {
        let pm = PackageManagerFacade::new(PackageManager::Unknown);
        let (prog, args) = pm.build_install_cmd(&["git"]);
        assert_eq!(prog, "echo");
        assert!(args.contains(&"unknown package manager".to_string()));
    }

    // =====================================================================
    // VAL-INFRA-006: Package manager supports dry-run mode
    // =====================================================================

    #[test]
    fn test_dry_run_install_returns_result_without_executing() {
        let pm = PackageManagerFacade::new(PackageManager::Apt).with_dry_run(true);
        let result = pm.install(&["git", "cmake"]).unwrap();
        assert!(result.is_some());
        let dry = result.unwrap();
        assert!(dry.to_command_string().contains("sudo apt"));
        assert!(dry.to_command_string().contains("install"));
        assert!(dry.description.contains("git"));
        assert!(dry.description.contains("cmake"));
    }

    #[test]
    fn test_dry_run_update_indices() {
        let pm = PackageManagerFacade::new(PackageManager::Apt).with_dry_run(true);
        let result = pm.update_indices().unwrap();
        assert!(result.is_some());
        let dry = result.unwrap();
        assert!(dry.to_command_string().contains("sudo apt update"));
    }

    #[test]
    fn test_dry_run_remove() {
        let pm = PackageManagerFacade::new(PackageManager::Pacman).with_dry_run(true);
        let result = pm.remove(&["nano"]).unwrap();
        assert!(result.is_some());
        let dry = result.unwrap();
        assert!(dry.to_command_string().contains("sudo pacman"));
        assert!(dry.to_command_string().contains("-R"));
    }

    #[test]
    fn test_dry_run_is_installed_returns_false() {
        let pm = PackageManagerFacade::new(PackageManager::Apt).with_dry_run(true);
        // Even if the package is installed, dry-run should return false
        let result = pm.is_installed("bash").unwrap();
        assert!(!result);
    }

    #[test]
    fn test_dry_run_search_returns_none() {
        let pm = PackageManagerFacade::new(PackageManager::Apt).with_dry_run(true);
        let result = pm.search("python").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_dry_run_info_returns_none() {
        let pm = PackageManagerFacade::new(PackageManager::Apt).with_dry_run(true);
        let result = pm.info("python3").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_dry_run_display_format() {
        let pm = PackageManagerFacade::new(PackageManager::Apt).with_dry_run(true);
        let result = pm.install(&["git"]).unwrap().unwrap();
        let display = format!("{}", result);
        assert!(display.starts_with("[DRY-RUN]"));
        assert!(display.contains("sudo apt"));
    }

    #[test]
    fn test_dry_run_command_string_format() {
        let dry = DryRunResult {
            program: "sudo apt".to_string(),
            args: vec!["install".to_string(), "-y".to_string(), "git".to_string()],
            description: "install git".to_string(),
        };
        assert_eq!(dry.to_command_string(), "sudo apt install -y git");
    }

    // --- Error cases ---

    #[test]
    fn test_install_empty_packages_errors() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let result = pm.install(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_empty_packages_errors() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let result = pm.remove(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_purge_empty_packages_errors() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        let result = pm.purge(&[]);
        assert!(result.is_err());
    }

    // --- PackageOperation Display ---

    #[test]
    fn test_package_operation_display() {
        assert_eq!(format!("{}", PackageOperation::Install), "install");
        assert_eq!(format!("{}", PackageOperation::Remove), "remove");
        assert_eq!(format!("{}", PackageOperation::Purge), "purge");
        assert_eq!(format!("{}", PackageOperation::Update), "update");
        assert_eq!(format!("{}", PackageOperation::Upgrade), "upgrade");
        assert_eq!(format!("{}", PackageOperation::Search), "search");
        assert_eq!(format!("{}", PackageOperation::Info), "info");
        assert_eq!(format!("{}", PackageOperation::IsInstalled), "is_installed");
        assert_eq!(
            format!("{}", PackageOperation::ListInstalled),
            "list_installed"
        );
    }

    // --- Accessors ---

    #[test]
    fn test_accessors() {
        let pm = PackageManagerFacade::new(PackageManager::Dnf).with_dry_run(true);
        assert_eq!(pm.package_manager(), PackageManager::Dnf);
        assert!(pm.is_dry_run());
    }

    #[test]
    fn test_dry_run_default_false() {
        let pm = PackageManagerFacade::new(PackageManager::Apt);
        assert!(!pm.is_dry_run());
    }
}
