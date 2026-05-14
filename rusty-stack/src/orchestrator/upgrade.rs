//! Upgrade command implementation — download, verify, swap, and rollback.
//!
//! This module implements the `rusty upgrade` flow:
//! 1. Check manifest version compatibility (runtime vs. manifest `min_runtime_version`)
//! 2. Download new binary from release URL
//! 3. Verify binary integrity (SHA-256 checksum against published value)
//! 4. Swap binary with backup of previous version
//! 5. Run smoke test to verify the new binary works
//! 6. Rollback on smoke test failure
//!
//! Supports interactive confirmation and non-interactive `--yes` mode.
//! Preserves cached remote manifest across upgrades.
//! Operationally separate from `rusty update`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Version information for the current binary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VersionInfo {
    pub version: String,
    pub schema_version: u32,
}

/// Metadata about an available upgrade target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReleaseInfo {
    pub version: String,
    pub download_url: String,
    pub checksum: String,
    /// Minimum runtime version required by this release.
    pub min_runtime_version: String,
    /// Schema version shipped with this release's bundled baseline manifest.
    pub schema_version: u32,
}

/// Result of a successful upgrade operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpgradeResult {
    pub previous_version: String,
    pub new_version: String,
    pub status: UpgradeStatus,
    /// Path where the previous binary was backed up.
    pub backup_path: PathBuf,
}

/// Status of an upgrade attempt.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UpgradeStatus {
    /// Upgrade completed successfully.
    Success,
    /// Upgrade failed and was rolled back.
    RolledBack,
    /// Upgrade was refused due to compatibility issues.
    Refused,
}

/// Options controlling upgrade behavior.
#[derive(Debug, Clone, Default)]
pub struct UpgradeOptions {
    /// Skip interactive confirmation prompts.
    pub non_interactive: bool,
    /// Custom binary path (defaults to current executable).
    pub binary_path: Option<PathBuf>,
    /// Custom backup directory (defaults to `~/.mlstack/backups`).
    pub backup_dir: Option<PathBuf>,
    /// Custom cached manifest path (defaults to `~/.mlstack/cache/remote_manifest.json`).
    pub cached_manifest_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during the upgrade process.
#[derive(Debug, Clone, PartialEq)]
pub enum UpgradeError {
    /// The target release requires a newer runtime than the current binary.
    IncompatibleRuntime { current: String, required: String },
    /// The current binary is too old to parse the manifest format.
    RuntimeTooOld {
        current_schema: u32,
        manifest_schema: u32,
    },
    /// Binary integrity check failed (checksum mismatch).
    IntegrityCheckFailed { expected: String, actual: String },
    /// The smoke test failed after swap.
    SmokeTestFailed { reason: String },
    /// Download failed.
    DownloadFailed { reason: String },
    /// User declined the upgrade in interactive mode.
    Declined,
    /// I/O error during file operations.
    IoError { path: String, reason: String },
}

impl std::fmt::Display for UpgradeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpgradeError::IncompatibleRuntime { current, required } => {
                write!(
                    f,
                    "incompatible runtime: current {current} does not meet required {required}"
                )
            }
            UpgradeError::RuntimeTooOld {
                current_schema,
                manifest_schema,
            } => {
                write!(
                    f,
                    "runtime too old: current schema v{current_schema} cannot parse manifest schema v{manifest_schema}; manual upgrade required"
                )
            }
            UpgradeError::IntegrityCheckFailed { expected, actual } => {
                write!(
                    f,
                    "integrity check failed: expected checksum {expected}, got {actual}"
                )
            }
            UpgradeError::SmokeTestFailed { reason } => {
                write!(f, "smoke test failed: {reason}")
            }
            UpgradeError::DownloadFailed { reason } => {
                write!(f, "download failed: {reason}")
            }
            UpgradeError::Declined => {
                write!(f, "upgrade declined by user")
            }
            UpgradeError::IoError { path, reason } => {
                write!(f, "I/O error at {path}: {reason}")
            }
        }
    }
}

impl std::error::Error for UpgradeError {}

// ---------------------------------------------------------------------------
// Core upgrade logic (trait-based for testability)
// ---------------------------------------------------------------------------

/// Trait for release discovery. Implementations provide release info from
/// a remote source or test fixture.
pub trait ReleaseProvider {
    /// Fetch the latest available release information.
    fn fetch_latest_release(&self) -> std::result::Result<ReleaseInfo, UpgradeError>;
}

/// Trait for binary download. Implementations provide the actual download
/// or test fixture data.
pub trait BinaryDownloader {
    /// Download the binary from the given URL and return its bytes.
    fn download(&self, url: &str) -> std::result::Result<Vec<u8>, UpgradeError>;
}

/// Trait for smoke-testing a binary. Implementations run the new binary
/// and check that it starts correctly.
pub trait SmokeTester {
    /// Run a smoke test on the binary at the given path.
    /// Returns `Ok(())` if the binary works, `Err` with reason on failure.
    fn test(&self, binary_path: &Path) -> std::result::Result<(), UpgradeError>;
}

/// Trait for user interaction. Implementations provide interactive
/// confirmation or auto-accept in non-interactive mode.
pub trait UserInteractor {
    /// Ask the user to confirm the upgrade from `current` to `target`.
    /// Returns `true` if confirmed, `false` if declined.
    fn confirm_upgrade(&self, current: &str, target: &str) -> bool;
}

// ---------------------------------------------------------------------------
// Version comparison helpers
// ---------------------------------------------------------------------------

/// Parse a version string like "1.2.3" or "1.2.3-rc1" into (major, minor, patch).
fn parse_version_parts(version: &str) -> Option<(u32, u32, u32)> {
    let base = version.split('-').next()?;
    let parts: Vec<&str> = base.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    Some((
        parts[0].parse().ok()?,
        parts[1].parse().ok()?,
        parts[2].parse().ok()?,
    ))
}

/// Check if `current` version satisfies the `required` minimum version.
/// Returns `Ok(())` if compatible, `Err(UpgradeError)` if not.
pub fn check_version_compatibility(
    current: &str,
    required: &str,
) -> std::result::Result<(), UpgradeError> {
    let cur = parse_version_parts(current).unwrap_or((0, 0, 0));
    let req = parse_version_parts(required).unwrap_or((0, 0, 0));

    if cur < req {
        return Err(UpgradeError::IncompatibleRuntime {
            current: current.to_string(),
            required: required.to_string(),
        });
    }
    Ok(())
}

/// Check if the current runtime's schema version can handle a manifest
/// with the given schema version.
pub fn check_schema_compatibility(
    current_schema: u32,
    manifest_schema: u32,
) -> std::result::Result<(), UpgradeError> {
    // The runtime can only parse manifests at or below its own schema version.
    // Newer manifest schemas require a runtime upgrade first.
    if current_schema < manifest_schema {
        return Err(UpgradeError::RuntimeTooOld {
            current_schema,
            manifest_schema,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Checksum verification
// ---------------------------------------------------------------------------

/// Compute the SHA-256 checksum of a byte slice.
pub fn compute_checksum(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    result
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

/// Verify that `data` matches the expected SHA-256 `checksum`.
pub fn verify_integrity(
    data: &[u8],
    expected_checksum: &str,
) -> std::result::Result<(), UpgradeError> {
    let actual = compute_checksum(data);
    if actual == expected_checksum {
        Ok(())
    } else {
        Err(UpgradeError::IntegrityCheckFailed {
            expected: expected_checksum.to_string(),
            actual,
        })
    }
}

// ---------------------------------------------------------------------------
// File operations
// ---------------------------------------------------------------------------

/// Get the default backup directory path.
pub fn default_backup_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".mlstack")
        .join("backups")
}

/// Get the default cached manifest path.
pub fn default_cached_manifest_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".mlstack")
        .join("cache")
        .join("remote_manifest.json")
}

/// Get the current binary path.
pub fn current_binary_path() -> Result<PathBuf> {
    std::env::current_exe().context("failed to determine current executable path")
}

/// Create a backup of the current binary.
///
/// Copies the binary at `source` to `backup_dir` with a versioned filename.
/// Returns the path to the backup file.
pub fn create_backup(source: &Path, backup_dir: &Path, version: &str) -> Result<PathBuf> {
    fs::create_dir_all(backup_dir).with_context(|| {
        format!(
            "failed to create backup directory: {}",
            backup_dir.display()
        )
    })?;

    let backup_name = format!("rusty-stack-{version}.bak");
    let backup_path = backup_dir.join(&backup_name);

    fs::copy(source, &backup_path).with_context(|| {
        format!(
            "failed to backup binary from {} to {}",
            source.display(),
            backup_path.display()
        )
    })?;

    Ok(backup_path)
}

/// Swap the old binary with the new one.
///
/// 1. Rename the current binary to a `.old` temporary file
/// 2. Write the new binary to the original path
/// 3. Set executable permissions
/// 4. Remove the `.old` temporary file
pub fn swap_binary(binary_path: &Path, new_data: &[u8]) -> Result<()> {
    let old_path = binary_path.with_extension("old");

    // Rename current binary to .old
    if binary_path.exists() {
        fs::rename(binary_path, &old_path).with_context(|| {
            format!(
                "failed to rename {} to {}",
                binary_path.display(),
                old_path.display()
            )
        })?;
    }

    // Write new binary
    let write_result = (|| -> Result<()> {
        let mut f = fs::File::create(binary_path)
            .with_context(|| format!("failed to create new binary at {}", binary_path.display()))?;
        f.write_all(new_data)
            .with_context(|| format!("failed to write new binary at {}", binary_path.display()))?;

        // Set executable permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(binary_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(binary_path, perms)?;
        }

        Ok(())
    })();

    if let Err(e) = write_result {
        // Attempt to restore the old binary on write failure
        if old_path.exists() {
            let _ = fs::rename(&old_path, binary_path);
        }
        return Err(e);
    }

    // Clean up the old temporary file
    if old_path.exists() {
        let _ = fs::remove_file(&old_path);
    }

    Ok(())
}

/// Rollback to a previous binary version.
///
/// Restores the backup binary to the binary path.
pub fn rollback(binary_path: &Path, backup_path: &Path) -> Result<()> {
    fs::copy(backup_path, binary_path).with_context(|| {
        format!(
            "failed to rollback binary from {} to {}",
            backup_path.display(),
            binary_path.display()
        )
    })?;

    // Set executable permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(binary_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(binary_path, perms)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Cached manifest preservation
// ---------------------------------------------------------------------------

/// Read the cached remote manifest bytes if present.
pub fn read_cached_manifest(path: &Path) -> Option<Vec<u8>> {
    fs::read(path).ok()
}

/// Write the cached remote manifest bytes back (preservation after upgrade).
pub fn write_cached_manifest(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create cache directory: {}", parent.display()))?;
    }
    fs::write(path, data)
        .with_context(|| format!("failed to write cached manifest to {}", path.display()))
}

// ---------------------------------------------------------------------------
// Full upgrade orchestration
// ---------------------------------------------------------------------------

/// Run the full upgrade flow.
///
/// This is the main entry point for the upgrade command. It:
/// 1. Checks version compatibility
/// 2. Fetches release info
/// 3. Optionally confirms with user
/// 4. Downloads the new binary
/// 5. Verifies integrity
/// 6. Preserves cached manifest
/// 7. Creates backup
/// 8. Swaps binary
/// 9. Runs smoke test
/// 10. Rolls back on failure
pub fn run_upgrade(
    current_version: &VersionInfo,
    options: &UpgradeOptions,
    release_provider: &dyn ReleaseProvider,
    downloader: &dyn BinaryDownloader,
    smoke_tester: &dyn SmokeTester,
    interactor: &dyn UserInteractor,
) -> std::result::Result<UpgradeResult, UpgradeError> {
    // Step 1: Fetch latest release info
    let release = release_provider.fetch_latest_release()?;

    // Step 2: Check manifest version compatibility
    check_schema_compatibility(current_version.schema_version, release.schema_version)?;

    // Step 3: Check runtime version compatibility
    check_version_compatibility(&current_version.version, &release.min_runtime_version)?;

    // Step 4: Interactive confirmation (if applicable)
    if !options.non_interactive {
        let confirmed = interactor.confirm_upgrade(&current_version.version, &release.version);
        if !confirmed {
            return Err(UpgradeError::Declined);
        }
    }

    // Step 5: Download new binary
    let new_binary_data = downloader.download(&release.download_url)?;

    // Step 6: Verify integrity
    verify_integrity(&new_binary_data, &release.checksum)?;

    // Step 7: Preserve cached manifest
    let cached_manifest_path = options
        .cached_manifest_path
        .clone()
        .unwrap_or_else(default_cached_manifest_path);
    let cached_manifest_data = read_cached_manifest(&cached_manifest_path);

    // Step 8: Determine binary path and backup directory
    let binary_path = options
        .binary_path
        .clone()
        .unwrap_or_else(|| current_binary_path().unwrap_or_else(|_| PathBuf::from("rusty-stack")));
    let backup_dir = options
        .backup_dir
        .clone()
        .unwrap_or_else(default_backup_dir);

    // Step 9: Create backup
    let backup_path =
        create_backup(&binary_path, &backup_dir, &current_version.version).map_err(|e| {
            UpgradeError::IoError {
                path: binary_path.display().to_string(),
                reason: e.to_string(),
            }
        })?;

    // Step 10: Swap binary
    swap_binary(&binary_path, &new_binary_data).map_err(|e| UpgradeError::IoError {
        path: binary_path.display().to_string(),
        reason: e.to_string(),
    })?;

    // Step 11: Run smoke test
    match smoke_tester.test(&binary_path) {
        Ok(()) => {
            // Success — restore cached manifest
            if let Some(data) = cached_manifest_data {
                let _ = write_cached_manifest(&cached_manifest_path, &data);
            }

            Ok(UpgradeResult {
                previous_version: current_version.version.clone(),
                new_version: release.version.clone(),
                status: UpgradeStatus::Success,
                backup_path,
            })
        }
        Err(smoke_err) => {
            // Rollback on smoke test failure
            let _ = rollback(&binary_path, &backup_path);

            // Restore cached manifest after rollback
            if let Some(data) = cached_manifest_data {
                let _ = write_cached_manifest(&cached_manifest_path, &data);
            }

            Err(UpgradeError::SmokeTestFailed {
                reason: smoke_err.to_string(),
            })
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    // ---- Version compatibility tests (VAL-UPGR-002, VAL-UPGR-003) ----

    #[test]
    fn test_version_compat_equal_versions() {
        assert!(check_version_compatibility("1.0.0", "1.0.0").is_ok());
    }

    #[test]
    fn test_version_compat_newer_current() {
        assert!(check_version_compatibility("2.0.0", "1.0.0").is_ok());
    }

    #[test]
    fn test_version_compat_older_current_refused() {
        let result = check_version_compatibility("1.0.0", "2.0.0");
        assert!(result.is_err());
        match result.unwrap_err() {
            UpgradeError::IncompatibleRuntime { current, required } => {
                assert_eq!(current, "1.0.0");
                assert_eq!(required, "2.0.0");
            }
            other => panic!("expected IncompatibleRuntime, got {other:?}"),
        }
    }

    #[test]
    fn test_version_compat_minor_older_refused() {
        let result = check_version_compatibility("1.0.0", "1.1.0");
        assert!(result.is_err());
    }

    #[test]
    fn test_version_compat_patch_newer_ok() {
        assert!(check_version_compatibility("1.0.1", "1.0.0").is_ok());
    }

    #[test]
    fn test_schema_compat_equal() {
        assert!(check_schema_compatibility(2, 2).is_ok());
    }

    #[test]
    fn test_schema_compat_newer_runtime_ok() {
        assert!(check_schema_compatibility(3, 2).is_ok());
    }

    #[test]
    fn test_schema_compat_older_runtime_refused() {
        let result = check_schema_compatibility(1, 3);
        assert!(result.is_err());
        match result.unwrap_err() {
            UpgradeError::RuntimeTooOld {
                current_schema,
                manifest_schema,
            } => {
                assert_eq!(current_schema, 1);
                assert_eq!(manifest_schema, 3);
            }
            other => panic!("expected RuntimeTooOld, got {other:?}"),
        }
    }

    // ---- Checksum verification tests (VAL-UPGR-010) ----

    #[test]
    fn test_checksum_valid() {
        let data = b"hello world";
        let expected = compute_checksum(data);
        assert!(verify_integrity(data, &expected).is_ok());
    }

    #[test]
    fn test_checksum_tampered_binary_rejected() {
        let data = b"original binary content";
        let expected = compute_checksum(data);
        let tampered = b"tampered binary content";
        let result = verify_integrity(tampered, &expected);
        assert!(result.is_err());
        match result.unwrap_err() {
            UpgradeError::IntegrityCheckFailed { .. } => {}
            other => panic!("expected IntegrityCheckFailed, got {other:?}"),
        }
    }

    #[test]
    fn test_checksum_empty_data() {
        let data = b"";
        let expected = compute_checksum(data);
        assert!(verify_integrity(data, &expected).is_ok());
    }

    // ---- Backup and rollback tests (VAL-UPGR-004, VAL-UPGR-012) ----

    #[test]
    fn test_backup_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("rusty-stack");
        fs::write(&source, b"binary content").unwrap();

        let backup_dir = dir.path().join("backups");
        let backup = create_backup(&source, &backup_dir, "0.1.0").unwrap();

        assert!(backup.exists());
        assert_eq!(fs::read(&backup).unwrap(), b"binary content".to_vec());
        assert_eq!(backup.file_name().unwrap(), "rusty-stack-0.1.0.bak");
    }

    #[test]
    fn test_backup_preserves_content() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("rusty-stack");
        let content = b"\x00\x01\x02\xff binary data with null bytes";
        fs::write(&source, content).unwrap();

        let backup_dir = dir.path().join("backups");
        let backup = create_backup(&source, &backup_dir, "1.2.3").unwrap();

        let backup_content = fs::read(&backup).unwrap();
        assert_eq!(backup_content, content.to_vec());
    }

    #[test]
    fn test_rollback_restores_binary() {
        let dir = tempfile::tempdir().unwrap();
        let binary_path = dir.path().join("rusty-stack");
        let backup_path = dir.path().join("rusty-stack-0.1.0.bak");

        let original_content = b"original binary";
        let new_content = b"new binary that fails smoke test";

        fs::write(&binary_path, new_content).unwrap();
        fs::write(&backup_path, original_content).unwrap();

        rollback(&binary_path, &backup_path).unwrap();

        let restored = fs::read(&binary_path).unwrap();
        assert_eq!(restored, original_content.to_vec());
    }

    // ---- Swap binary test (VAL-UPGR-004) ----

    #[test]
    fn test_swap_binary_replaces_content() {
        let dir = tempfile::tempdir().unwrap();
        let binary_path = dir.path().join("rusty-stack");
        fs::write(&binary_path, b"old binary").unwrap();

        let new_data = b"new binary";
        swap_binary(&binary_path, new_data).unwrap();

        let content = fs::read(&binary_path).unwrap();
        assert_eq!(content, new_data.to_vec());
    }

    #[test]
    fn test_swap_binary_creates_if_missing() {
        let dir = tempfile::tempdir().unwrap();
        let binary_path = dir.path().join("rusty-stack");
        // Don't create the file — swap should still work

        let new_data = b"brand new binary";
        swap_binary(&binary_path, new_data).unwrap();

        let content = fs::read(&binary_path).unwrap();
        assert_eq!(content, new_data.to_vec());
    }

    // ---- Cached manifest preservation tests (VAL-UPGR-006) ----

    #[test]
    fn test_cached_manifest_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache").join("remote_manifest.json");
        let data = b"{\"schema_version\": 2, \"components\": []}";

        write_cached_manifest(&path, data).unwrap();
        let read_data = read_cached_manifest(&path).unwrap();

        assert_eq!(read_data, data.to_vec());
    }

    #[test]
    fn test_cached_manifest_missing_returns_none() {
        let path = PathBuf::from("/nonexistent/path/manifest.json");
        assert!(read_cached_manifest(&path).is_none());
    }

    #[test]
    fn test_cached_manifest_preserves_exact_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.json");

        // Write manifest with specific formatting
        let original = b"{\n  \"schema_version\": 2,\n  \"components\": []\n}";
        write_cached_manifest(&path, original).unwrap();

        let restored = read_cached_manifest(&path).unwrap();
        assert_eq!(restored, original.to_vec());
    }

    // ---- Full upgrade flow tests (VAL-UPGR-001, VAL-UPGR-004, VAL-UPGR-007, VAL-UPGR-011) ----

    /// A mock release provider for testing.
    struct MockReleaseProvider {
        release: std::result::Result<ReleaseInfo, UpgradeError>,
    }

    impl ReleaseProvider for MockReleaseProvider {
        fn fetch_latest_release(&self) -> std::result::Result<ReleaseInfo, UpgradeError> {
            self.release.clone()
        }
    }

    /// A mock binary downloader for testing.
    struct MockDownloader {
        data: std::result::Result<Vec<u8>, UpgradeError>,
    }

    impl BinaryDownloader for MockDownloader {
        fn download(&self, _url: &str) -> std::result::Result<Vec<u8>, UpgradeError> {
            self.data.clone()
        }
    }

    /// A mock smoke tester for testing.
    struct MockSmokeTester {
        result: std::result::Result<(), UpgradeError>,
    }

    impl SmokeTester for MockSmokeTester {
        fn test(&self, _binary_path: &Path) -> std::result::Result<(), UpgradeError> {
            self.result.clone()
        }
    }

    /// A mock user interactor for testing.
    struct MockInteractor {
        confirmed: bool,
    }

    impl UserInteractor for MockInteractor {
        fn confirm_upgrade(&self, _current: &str, _target: &str) -> bool {
            self.confirmed
        }
    }

    /// A smoke tester that records whether it was called.
    #[allow(dead_code)]
    struct RecordingSmokeTester {
        called: AtomicBool,
        result: std::result::Result<(), UpgradeError>,
    }

    impl SmokeTester for RecordingSmokeTester {
        fn test(&self, _binary_path: &Path) -> std::result::Result<(), UpgradeError> {
            self.called.store(true, Ordering::SeqCst);
            self.result.clone()
        }
    }

    fn make_release_info(version: &str, min_runtime: &str, schema: u32) -> ReleaseInfo {
        ReleaseInfo {
            version: version.to_string(),
            download_url: format!("https://example.com/rusty-stack-{version}"),
            checksum: String::new(), // Will be set by test
            min_runtime_version: min_runtime.to_string(),
            schema_version: schema,
        }
    }

    #[test]
    fn test_full_upgrade_success() {
        let dir = tempfile::tempdir().unwrap();
        let binary_path = dir.path().join("rusty-stack");
        let backup_dir = dir.path().join("backups");
        let cache_dir = dir.path().join("cache");
        let cached_manifest = cache_dir.join("remote_manifest.json");

        // Set up initial binary
        fs::write(&binary_path, b"old binary v0.1.0").unwrap();

        // Set up cached manifest
        let manifest_data = b"{\"schema_version\": 2}";
        write_cached_manifest(&cached_manifest, manifest_data).unwrap();

        let new_binary = b"new binary v0.2.0";
        let checksum = compute_checksum(new_binary);

        let mut release = make_release_info("0.2.0", "0.1.0", 2);
        release.checksum = checksum;

        let result = run_upgrade(
            &VersionInfo {
                version: "0.1.0".to_string(),
                schema_version: 2,
            },
            &UpgradeOptions {
                non_interactive: true,
                binary_path: Some(binary_path.clone()),
                backup_dir: Some(backup_dir.clone()),
                cached_manifest_path: Some(cached_manifest.clone()),
            },
            &MockReleaseProvider {
                release: Ok(release),
            },
            &MockDownloader {
                data: Ok(new_binary.to_vec()),
            },
            &MockSmokeTester { result: Ok(()) },
            &MockInteractor { confirmed: true },
        );

        assert!(result.is_ok(), "upgrade should succeed");
        let upgrade_result = result.unwrap();
        assert_eq!(upgrade_result.previous_version, "0.1.0");
        assert_eq!(upgrade_result.new_version, "0.2.0");
        assert_eq!(upgrade_result.status, UpgradeStatus::Success);
        assert!(upgrade_result.backup_path.exists());

        // Binary was replaced
        let content = fs::read(&binary_path).unwrap();
        assert_eq!(content, new_binary.to_vec());

        // Cached manifest preserved
        let restored_manifest = read_cached_manifest(&cached_manifest).unwrap();
        assert_eq!(restored_manifest, manifest_data.to_vec());
    }

    #[test]
    fn test_upgrade_rollback_on_smoke_test_failure() {
        let dir = tempfile::tempdir().unwrap();
        let binary_path = dir.path().join("rusty-stack");
        let backup_dir = dir.path().join("backups");

        let original_content = b"original binary v0.1.0";
        fs::write(&binary_path, original_content).unwrap();

        let new_binary = b"broken new binary";
        let checksum = compute_checksum(new_binary);

        let mut release = make_release_info("0.2.0", "0.1.0", 2);
        release.checksum = checksum;

        let result = run_upgrade(
            &VersionInfo {
                version: "0.1.0".to_string(),
                schema_version: 2,
            },
            &UpgradeOptions {
                non_interactive: true,
                binary_path: Some(binary_path.clone()),
                backup_dir: Some(backup_dir),
                cached_manifest_path: None,
            },
            &MockReleaseProvider {
                release: Ok(release),
            },
            &MockDownloader {
                data: Ok(new_binary.to_vec()),
            },
            &MockSmokeTester {
                result: Err(UpgradeError::SmokeTestFailed {
                    reason: "binary exited with code 1".to_string(),
                }),
            },
            &MockInteractor { confirmed: true },
        );

        assert!(result.is_err(), "upgrade should fail on smoke test");
        match result.unwrap_err() {
            UpgradeError::SmokeTestFailed { reason } => {
                assert!(reason.contains("binary exited with code 1"));
            }
            other => panic!("expected SmokeTestFailed, got {other:?}"),
        }

        // Original binary should be restored
        let content = fs::read(&binary_path).unwrap();
        assert_eq!(content, original_content.to_vec());
    }

    #[test]
    fn test_upgrade_refused_incompatible_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let binary_path = dir.path().join("rusty-stack");
        fs::write(&binary_path, b"old binary").unwrap();

        let release = make_release_info("2.0.0", "2.0.0", 2);

        let result = run_upgrade(
            &VersionInfo {
                version: "1.0.0".to_string(),
                schema_version: 2,
            },
            &UpgradeOptions {
                non_interactive: true,
                binary_path: Some(binary_path.clone()),
                backup_dir: Some(dir.path().join("backups")),
                cached_manifest_path: None,
            },
            &MockReleaseProvider {
                release: Ok(release),
            },
            &MockDownloader { data: Ok(vec![]) },
            &MockSmokeTester { result: Ok(()) },
            &MockInteractor { confirmed: true },
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            UpgradeError::IncompatibleRuntime { .. } => {}
            other => panic!("expected IncompatibleRuntime, got {other:?}"),
        }

        // Binary should be unchanged
        let content = fs::read(&binary_path).unwrap();
        assert_eq!(content, b"old binary".to_vec());
    }

    #[test]
    fn test_upgrade_refused_runtime_too_old_for_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let binary_path = dir.path().join("rusty-stack");
        fs::write(&binary_path, b"old binary v1 schema").unwrap();

        let release = make_release_info("0.2.0", "0.1.0", 3);

        let result = run_upgrade(
            &VersionInfo {
                version: "0.1.0".to_string(),
                schema_version: 1,
            },
            &UpgradeOptions {
                non_interactive: true,
                binary_path: Some(binary_path.clone()),
                backup_dir: Some(dir.path().join("backups")),
                cached_manifest_path: None,
            },
            &MockReleaseProvider {
                release: Ok(release),
            },
            &MockDownloader { data: Ok(vec![]) },
            &MockSmokeTester { result: Ok(()) },
            &MockInteractor { confirmed: true },
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            UpgradeError::RuntimeTooOld {
                current_schema,
                manifest_schema,
            } => {
                assert_eq!(current_schema, 1);
                assert_eq!(manifest_schema, 3);
            }
            other => panic!("expected RuntimeTooOld, got {other:?}"),
        }
    }

    #[test]
    fn test_upgrade_declined_in_interactive_mode() {
        let dir = tempfile::tempdir().unwrap();
        let binary_path = dir.path().join("rusty-stack");
        let original_content = b"original binary";
        fs::write(&binary_path, original_content).unwrap();

        let new_binary = b"new binary";
        let checksum = compute_checksum(new_binary);

        let mut release = make_release_info("0.2.0", "0.1.0", 2);
        release.checksum = checksum;

        let result = run_upgrade(
            &VersionInfo {
                version: "0.1.0".to_string(),
                schema_version: 2,
            },
            &UpgradeOptions {
                non_interactive: false, // Interactive mode
                binary_path: Some(binary_path.clone()),
                backup_dir: Some(dir.path().join("backups")),
                cached_manifest_path: None,
            },
            &MockReleaseProvider {
                release: Ok(release),
            },
            &MockDownloader {
                data: Ok(new_binary.to_vec()),
            },
            &MockSmokeTester { result: Ok(()) },
            &MockInteractor { confirmed: false }, // User declines
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            UpgradeError::Declined => {}
            other => panic!("expected Declined, got {other:?}"),
        }

        // Binary should be unchanged
        let content = fs::read(&binary_path).unwrap();
        assert_eq!(content, original_content.to_vec());
    }

    #[test]
    fn test_upgrade_integrity_failure_preserves_binary() {
        let dir = tempfile::tempdir().unwrap();
        let binary_path = dir.path().join("rusty-stack");
        let original_content = b"original binary";
        fs::write(&binary_path, original_content).unwrap();

        let new_binary = b"corrupted new binary";

        let mut release = make_release_info("0.2.0", "0.1.0", 2);
        release.checksum = "wrong_checksum_value".to_string();

        let result = run_upgrade(
            &VersionInfo {
                version: "0.1.0".to_string(),
                schema_version: 2,
            },
            &UpgradeOptions {
                non_interactive: true,
                binary_path: Some(binary_path.clone()),
                backup_dir: Some(dir.path().join("backups")),
                cached_manifest_path: None,
            },
            &MockReleaseProvider {
                release: Ok(release),
            },
            &MockDownloader {
                data: Ok(new_binary.to_vec()),
            },
            &MockSmokeTester { result: Ok(()) },
            &MockInteractor { confirmed: true },
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            UpgradeError::IntegrityCheckFailed { .. } => {}
            other => panic!("expected IntegrityCheckFailed, got {other:?}"),
        }

        // Binary should be unchanged (integrity check happens before swap)
        let content = fs::read(&binary_path).unwrap();
        assert_eq!(content, original_content.to_vec());
    }

    // ---- Version parsing edge cases ----

    #[test]
    fn test_parse_version_parts_standard() {
        assert_eq!(parse_version_parts("1.2.3"), Some((1, 2, 3)));
    }

    #[test]
    fn test_parse_version_parts_prerelease() {
        assert_eq!(parse_version_parts("1.2.3-rc1"), Some((1, 2, 3)));
    }

    #[test]
    fn test_parse_version_parts_invalid() {
        assert_eq!(parse_version_parts("invalid"), None);
        assert_eq!(parse_version_parts("1.2"), None);
        assert_eq!(parse_version_parts(""), None);
    }

    // ---- UpgradeResult JSON output test (VAL-UPGR-009) ----

    #[test]
    fn test_upgrade_result_serializes_to_json() {
        let result = UpgradeResult {
            previous_version: "0.1.0".to_string(),
            new_version: "0.2.0".to_string(),
            status: UpgradeStatus::Success,
            backup_path: PathBuf::from("/home/user/.mlstack/backups/rusty-stack-0.1.0.bak"),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"previous_version\":\"0.1.0\""));
        assert!(json.contains("\"new_version\":\"0.2.0\""));
        assert!(json.contains("\"status\":\"success\""));

        // Roundtrip
        let back: UpgradeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back, result);
    }

    #[test]
    fn test_upgrade_result_rolled_back_serializes() {
        let result = UpgradeResult {
            previous_version: "0.1.0".to_string(),
            new_version: "0.2.0".to_string(),
            status: UpgradeStatus::RolledBack,
            backup_path: PathBuf::from("/tmp/backup"),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"status\":\"rolled_back\""));
    }

    #[test]
    fn test_upgrade_result_refused_serializes() {
        let result = UpgradeResult {
            previous_version: "0.1.0".to_string(),
            new_version: "0.2.0".to_string(),
            status: UpgradeStatus::Refused,
            backup_path: PathBuf::from("/tmp/backup"),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"status\":\"refused\""));
    }
}
