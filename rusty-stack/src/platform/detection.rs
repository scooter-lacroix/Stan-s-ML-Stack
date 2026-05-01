//! Linux distribution detection and package manager identification.
//!
//! Parses `/etc/os-release` (primary), falls back to `/etc/lsb-release`,
//! `/etc/arch-release`, and `/etc/debian_version`. Detects the distro family
//! and the appropriate package manager.
//!
//! # Validation Assertions
//!
//! - **VAL-PLAT-007**: Distro detection identifies correct Linux distribution
//! - **VAL-PLAT-008**: Package manager detection matches distro family,
//!   `MLSTACK_PKG_MANAGER` env var overrides auto-detection

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// ===========================================================================
// Public Types
// ===========================================================================

/// Linux distribution family classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistroFamily {
    Debian,
    Arch,
    Rhel,
    Suse,
    Unknown,
}

impl DistroFamily {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            DistroFamily::Debian => "debian",
            DistroFamily::Arch => "arch",
            DistroFamily::Rhel => "rhel",
            DistroFamily::Suse => "suse",
            DistroFamily::Unknown => "unknown",
        }
    }
}

/// Known package managers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PackageManager {
    Apt,
    Pacman,
    Dnf,
    Yum,
    Zypper,
    Unknown,
}

impl PackageManager {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            PackageManager::Apt => "apt",
            PackageManager::Pacman => "pacman",
            PackageManager::Dnf => "dnf",
            PackageManager::Yum => "yum",
            PackageManager::Zypper => "zypper",
            PackageManager::Unknown => "unknown",
        }
    }
}

/// Complete distribution information detected from the system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistroInfo {
    /// Distribution ID (e.g., "ubuntu", "arch", "fedora").
    pub id: String,
    /// Full distribution name (e.g., "Ubuntu 24.04 LTS").
    pub name: String,
    /// Distribution version (e.g., "24.04", "rolling", "42").
    pub version: String,
    /// Distribution codename (e.g., "noble", "bookworm").
    pub codename: String,
    /// Distribution family.
    pub family: DistroFamily,
    /// Detected package manager.
    pub pkg_manager: PackageManager,
    /// The `ID_LIKE` field from `/etc/os-release`.
    pub id_like: String,
}

impl Default for DistroInfo {
    fn default() -> Self {
        Self {
            id: "unknown".to_string(),
            name: "Unknown Linux".to_string(),
            version: "unknown".to_string(),
            codename: String::new(),
            family: DistroFamily::Unknown,
            pkg_manager: PackageManager::Unknown,
            id_like: String::new(),
        }
    }
}

// ===========================================================================
// Public API
// ===========================================================================

/// Detect the Linux distribution by parsing system files.
///
/// Follows a priority chain:
/// 1. `/etc/os-release` (standard, preferred)
/// 2. `/etc/lsb-release` (older Ubuntu/Debian)
/// 3. `/etc/arch-release` (Arch Linux marker file)
/// 4. `/etc/debian_version` (Debian fallback)
///
/// After identifying the distribution, determines the family and package
/// manager. The `MLSTACK_PKG_MANAGER` environment variable overrides
/// auto-detected package manager.
pub fn detect_distribution() -> DistroInfo {
    let mut info = DistroInfo::default();

    // Method 1: Parse /etc/os-release
    if parse_os_release(&mut info) {
        finalize_distro_info(&mut info);
        return info;
    }

    // Method 2: Fallback to /etc/lsb-release
    if parse_lsb_release(&mut info) {
        finalize_distro_info(&mut info);
        return info;
    }

    // Method 3: Arch Linux marker file
    if Path::new("/etc/arch-release").exists() {
        info.id = "arch".to_string();
        info.name = "Arch Linux".to_string();
        info.version = "rolling".to_string();
        finalize_distro_info(&mut info);
        return info;
    }

    // Method 4: Debian version file
    if Path::new("/etc/debian_version").exists() {
        info.id = "debian".to_string();
        info.name = "Debian GNU/Linux".to_string();
        if let Ok(version) = fs::read_to_string("/etc/debian_version") {
            info.version = version.trim().to_string();
        }
        finalize_distro_info(&mut info);
        return info;
    }

    // Last resort: unknown
    finalize_distro_info(&mut info);
    info
}

/// Detect the package manager from the distro family or environment override.
///
/// Checks `MLSTACK_PKG_MANAGER` env var first, then falls back to
/// family-based detection.
pub fn detect_package_manager(info: &DistroInfo) -> PackageManager {
    // Environment variable override
    if let Ok(pkg_mgr) = std::env::var("MLSTACK_PKG_MANAGER") {
        let pkg_mgr = pkg_mgr.trim().to_lowercase();
        if let Some(pm) = match_pkg_manager_str(&pkg_mgr) {
            return pm;
        }
    }

    // Family-based detection
    match info.family {
        DistroFamily::Debian => PackageManager::Apt,
        DistroFamily::Arch => PackageManager::Pacman,
        DistroFamily::Rhel => PackageManager::Dnf,
        DistroFamily::Suse => PackageManager::Zypper,
        DistroFamily::Unknown => PackageManager::Unknown,
    }
}

/// Parse a key-value file (like `/etc/os-release`) into a map.
///
/// Handles quoted and unquoted values, comments, and blank lines.
/// This is a public helper for testing with mock files.
pub fn parse_key_value_file(content: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim().to_string();
            let value = value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
            map.insert(key, value);
        }
    }
    map
}

// ===========================================================================
// Private Implementation
// ===========================================================================

/// Parse `/etc/os-release` and populate `DistroInfo`.
fn parse_os_release(info: &mut DistroInfo) -> bool {
    let content = match fs::read_to_string("/etc/os-release") {
        Ok(c) => c,
        Err(_) => return false,
    };

    let map = parse_key_value_file(&content);

    info.id = map.get("ID").cloned().unwrap_or_default().to_lowercase();
    info.name = map.get("NAME").cloned().unwrap_or_default();
    info.version = map.get("VERSION_ID").cloned().unwrap_or_default();
    info.codename = map.get("VERSION_CODENAME").cloned().unwrap_or_default();
    info.id_like = map.get("ID_LIKE").cloned().unwrap_or_default();

    // Handle rolling releases
    if info.version.is_empty() && is_rolling_distro(&info.id) {
        info.version = "rolling".to_string();
    }

    !info.id.is_empty()
}

/// Parse `/etc/lsb-release` and populate `DistroInfo`.
fn parse_lsb_release(info: &mut DistroInfo) -> bool {
    let content = match fs::read_to_string("/etc/lsb-release") {
        Ok(c) => c,
        Err(_) => return false,
    };

    let map = parse_key_value_file(&content);

    info.id = map
        .get("DISTRIB_ID")
        .cloned()
        .unwrap_or_default()
        .to_lowercase();
    info.name = map.get("DISTRIB_ID").cloned().unwrap_or_default();
    info.version = map.get("DISTRIB_RELEASE").cloned().unwrap_or_default();
    info.codename = map.get("DISTRIB_CODENAME").cloned().unwrap_or_default();

    !info.id.is_empty()
}

/// Determine the distro family from the distro ID and ID_LIKE fields.
fn classify_distro_family(id: &str, id_like: &str) -> DistroFamily {
    let id_lower = id.to_lowercase();
    let id_like_lower = id_like.to_lowercase();

    // Direct ID matches
    match id_lower.as_str() {
        "debian" | "ubuntu" | "linuxmint" | "pop" | "elementary" | "kali" | "mx" | "devuan"
        | "raspbian" => return DistroFamily::Debian,
        "arch" | "manjaro" | "cachyos" | "endeavouros" | "garuda" | "arco" | "artix"
        | "rebornos" => return DistroFamily::Arch,
        "fedora" | "rhel" | "centos" | "rocky" | "almalinux" | "ol" | "oracle" | "scientific"
        | "centos-stream" => return DistroFamily::Rhel,
        "opensuse-leap" | "opensuse-tumbleweed" | "opensuse" | "sles" | "suse" | "sle" => {
            return DistroFamily::Suse
        }
        _ => {}
    }

    // ID_LIKE matches
    if id_like_lower.contains("debian") || id_like_lower.contains("ubuntu") {
        return DistroFamily::Debian;
    }
    if id_like_lower.contains("arch") {
        return DistroFamily::Arch;
    }
    if id_like_lower.contains("fedora")
        || id_like_lower.contains("rhel")
        || id_like_lower.contains("centos")
    {
        return DistroFamily::Rhel;
    }
    if id_like_lower.contains("suse") {
        return DistroFamily::Suse;
    }

    DistroFamily::Unknown
}

/// Check if a distro ID is a known rolling release.
fn is_rolling_distro(id: &str) -> bool {
    matches!(
        id,
        "arch"
            | "cachyos"
            | "manjaro"
            | "endeavouros"
            | "garuda"
            | "arco"
            | "artix"
            | "rebornos"
            | "opensuse-tumbleweed"
    )
}

/// Match a package manager string to the enum.
fn match_pkg_manager_str(s: &str) -> Option<PackageManager> {
    match s {
        "apt" | "apt-get" => Some(PackageManager::Apt),
        "pacman" => Some(PackageManager::Pacman),
        "dnf" => Some(PackageManager::Dnf),
        "yum" => Some(PackageManager::Yum),
        "zypper" => Some(PackageManager::Zypper),
        _ => None,
    }
}

/// Finalize distro info: classify family and detect package manager.
fn finalize_distro_info(info: &mut DistroInfo) {
    info.family = classify_distro_family(&info.id, &info.id_like);
    info.pkg_manager = detect_package_manager(info);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // VAL-PLAT-007: Distro detection identifies correct Linux distribution
    // -----------------------------------------------------------------------

    #[test]
    fn test_distro_detection_parses_os_release() {
        // Create a mock os-release file
        let dir = tempfile::tempdir().unwrap();
        let os_release = dir.path().join("os-release");
        fs::write(
            &os_release,
            r#"NAME="Ubuntu"
VERSION="24.04 LTS (Noble Numbat)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 24.04 LTS"
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#,
        )
        .unwrap();

        let content = fs::read_to_string(&os_release).unwrap();
        let map = parse_key_value_file(&content);

        assert_eq!(map.get("ID"), Some(&"ubuntu".to_string()));
        assert_eq!(map.get("NAME"), Some(&"Ubuntu".to_string()));
        assert_eq!(map.get("VERSION_ID"), Some(&"24.04".to_string()));
        assert_eq!(map.get("VERSION_CODENAME"), Some(&"noble".to_string()));
        assert_eq!(map.get("ID_LIKE"), Some(&"debian".to_string()));
    }

    #[test]
    fn test_distro_detection_arch_linux() {
        let content = r#"NAME="Arch Linux"
PRETTY_NAME="Arch Linux"
ID=arch
BUILD_ID=rolling
"#;
        let map = parse_key_value_file(content);
        assert_eq!(map.get("ID"), Some(&"arch".to_string()));
        assert_eq!(map.get("NAME"), Some(&"Arch Linux".to_string()));
    }

    #[test]
    fn test_distro_detection_fedora() {
        let content = r#"NAME="Fedora Linux"
VERSION="41 (Workstation Edition)"
ID=fedora
VERSION_ID=41
ID_LIKE="rhel fedora"
"#;
        let map = parse_key_value_file(content);
        assert_eq!(map.get("ID"), Some(&"fedora".to_string()));
        assert_eq!(map.get("VERSION_ID"), Some(&"41".to_string()));
    }

    #[test]
    fn test_distro_detection_opensuse() {
        let content = r#"NAME="openSUSE Leap"
VERSION="15.5"
ID="opensuse-leap"
VERSION_ID="15.5"
ID_LIKE="suse opensuse"
"#;
        let map = parse_key_value_file(content);
        assert_eq!(map.get("ID"), Some(&"opensuse-leap".to_string()));
        assert_eq!(map.get("VERSION_ID"), Some(&"15.5".to_string()));
    }

    #[test]
    fn test_distro_detection_cachyos() {
        let content = r#"NAME="CachyOS"
ID=cachyos
ID_LIKE=arch
"#;
        let map = parse_key_value_file(content);
        assert_eq!(map.get("ID"), Some(&"cachyos".to_string()));
        assert_eq!(map.get("ID_LIKE"), Some(&"arch".to_string()));
    }

    #[test]
    fn test_distro_family_classification_debian() {
        assert_eq!(
            classify_distro_family("ubuntu", "debian"),
            DistroFamily::Debian
        );
        assert_eq!(classify_distro_family("debian", ""), DistroFamily::Debian);
        assert_eq!(
            classify_distro_family("linuxmint", "debian"),
            DistroFamily::Debian
        );
        assert_eq!(
            classify_distro_family("pop", "ubuntu debian"),
            DistroFamily::Debian
        );
        assert_eq!(
            classify_distro_family("kali", "debian"),
            DistroFamily::Debian
        );
    }

    #[test]
    fn test_distro_family_classification_arch() {
        assert_eq!(classify_distro_family("arch", ""), DistroFamily::Arch);
        assert_eq!(
            classify_distro_family("cachyos", "arch"),
            DistroFamily::Arch
        );
        assert_eq!(
            classify_distro_family("manjaro", "arch"),
            DistroFamily::Arch
        );
        assert_eq!(
            classify_distro_family("endeavouros", "arch"),
            DistroFamily::Arch
        );
    }

    #[test]
    fn test_distro_family_classification_rhel() {
        assert_eq!(
            classify_distro_family("fedora", "rhel fedora"),
            DistroFamily::Rhel
        );
        assert_eq!(
            classify_distro_family("rocky", "rhel centos"),
            DistroFamily::Rhel
        );
        assert_eq!(
            classify_distro_family("almalinux", "rhel centos"),
            DistroFamily::Rhel
        );
        assert_eq!(
            classify_distro_family("centos", "rhel fedora"),
            DistroFamily::Rhel
        );
    }

    #[test]
    fn test_distro_family_classification_suse() {
        assert_eq!(
            classify_distro_family("opensuse-leap", "suse opensuse"),
            DistroFamily::Suse
        );
        assert_eq!(
            classify_distro_family("opensuse-tumbleweed", "suse opensuse"),
            DistroFamily::Suse
        );
        assert_eq!(classify_distro_family("sles", "suse"), DistroFamily::Suse);
    }

    #[test]
    fn test_distro_family_classification_unknown() {
        assert_eq!(classify_distro_family("gentoo", ""), DistroFamily::Unknown);
        assert_eq!(classify_distro_family("alpine", ""), DistroFamily::Unknown);
    }

    #[test]
    fn test_rolling_distro_detection() {
        assert!(is_rolling_distro("arch"));
        assert!(is_rolling_distro("cachyos"));
        assert!(is_rolling_distro("manjaro"));
        assert!(is_rolling_distro("endeavouros"));
        assert!(is_rolling_distro("opensuse-tumbleweed"));
        assert!(!is_rolling_distro("ubuntu"));
        assert!(!is_rolling_distro("fedora"));
        assert!(!is_rolling_distro("debian"));
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-008: Package manager detection matches distro family
    // -----------------------------------------------------------------------

    #[test]
    fn test_package_manager_debian_family() {
        let info = DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            ..Default::default()
        };
        assert_eq!(detect_package_manager(&info), PackageManager::Apt);
    }

    #[test]
    fn test_package_manager_arch_family() {
        let info = DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            ..Default::default()
        };
        assert_eq!(detect_package_manager(&info), PackageManager::Pacman);
    }

    #[test]
    fn test_package_manager_rhel_family() {
        let info = DistroInfo {
            id: "fedora".to_string(),
            family: DistroFamily::Rhel,
            ..Default::default()
        };
        assert_eq!(detect_package_manager(&info), PackageManager::Dnf);
    }

    #[test]
    fn test_package_manager_suse_family() {
        let info = DistroInfo {
            id: "opensuse-leap".to_string(),
            family: DistroFamily::Suse,
            ..Default::default()
        };
        assert_eq!(detect_package_manager(&info), PackageManager::Zypper);
    }

    #[test]
    fn test_package_manager_unknown_family() {
        let info = DistroInfo {
            id: "alpine".to_string(),
            family: DistroFamily::Unknown,
            ..Default::default()
        };
        assert_eq!(detect_package_manager(&info), PackageManager::Unknown);
    }

    #[test]
    fn test_package_manager_env_override() {
        let saved = std::env::var("MLSTACK_PKG_MANAGER").ok();
        let info = DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            ..Default::default()
        };

        // Set override
        std::env::set_var("MLSTACK_PKG_MANAGER", "apt");
        assert_eq!(detect_package_manager(&info), PackageManager::Apt);

        // Restore original state
        match saved {
            Some(v) => std::env::set_var("MLSTACK_PKG_MANAGER", v),
            None => std::env::remove_var("MLSTACK_PKG_MANAGER"),
        }
    }

    #[test]
    fn test_match_pkg_manager_str() {
        assert_eq!(match_pkg_manager_str("apt"), Some(PackageManager::Apt));
        assert_eq!(match_pkg_manager_str("apt-get"), Some(PackageManager::Apt));
        assert_eq!(
            match_pkg_manager_str("pacman"),
            Some(PackageManager::Pacman)
        );
        assert_eq!(match_pkg_manager_str("dnf"), Some(PackageManager::Dnf));
        assert_eq!(match_pkg_manager_str("yum"), Some(PackageManager::Yum));
        assert_eq!(
            match_pkg_manager_str("zypper"),
            Some(PackageManager::Zypper)
        );
        assert_eq!(match_pkg_manager_str("unknown"), None);
    }

    // -----------------------------------------------------------------------
    // Real system detection (not mocked, validates against current system)
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_distribution_real_system() {
        let info = detect_distribution();
        // On this system (CachyOS/Arch), we should detect something
        assert!(
            !info.id.is_empty(),
            "Distro ID should be non-empty on a real system"
        );
        assert!(
            !info.name.is_empty(),
            "Distro name should be non-empty on a real system"
        );
        // Should have a valid family
        assert!(
            info.family != DistroFamily::Unknown || info.id == "unknown",
            "Should classify family for known distros"
        );
    }

    #[test]
    fn test_detect_distribution_real_system_distro_info_serde_roundtrip() {
        let info = detect_distribution();
        let json = serde_json::to_string(&info).unwrap();
        let back: DistroInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    // -----------------------------------------------------------------------
    // parse_key_value_file edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_key_value_file_empty() {
        let map = parse_key_value_file("");
        assert!(map.is_empty());
    }

    #[test]
    fn test_parse_key_value_file_comments_and_blanks() {
        let content = "# This is a comment\n\nID=arch\n# Another comment\nNAME=\"Arch Linux\"\n";
        let map = parse_key_value_file(content);
        assert_eq!(map.get("ID"), Some(&"arch".to_string()));
        assert_eq!(map.get("NAME"), Some(&"Arch Linux".to_string()));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_parse_key_value_file_single_quoted() {
        let content = "ID='ubuntu'\nNAME='Ubuntu'";
        let map = parse_key_value_file(content);
        assert_eq!(map.get("ID"), Some(&"ubuntu".to_string()));
        assert_eq!(map.get("NAME"), Some(&"Ubuntu".to_string()));
    }

    #[test]
    fn test_parse_key_value_file_unquoted() {
        let content = "ID=arch\nVERSION_ID=rolling";
        let map = parse_key_value_file(content);
        assert_eq!(map.get("ID"), Some(&"arch".to_string()));
        assert_eq!(map.get("VERSION_ID"), Some(&"rolling".to_string()));
    }
}
