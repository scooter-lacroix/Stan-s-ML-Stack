//! Windows ↔ WSL2 path translation bridge.
//!
//! Provides bidirectional path translation between Windows and WSL2 path
//! formats, handling drive letters, UNC paths, Unicode, and spaces.
//!
//! # Path Translation Rules
//!
//! **Windows → WSL2:**
//! - `C:\Users\alice` → `/mnt/c/Users/alice`
//! - `D:\data\file.txt` → `/mnt/d/data/file.txt`
//! - `\\wsl$\Ubuntu\home\alice` → `/home/alice` (UNC WSL path)
//!
//! **WSL2 → Windows:**
//! - `/home/alice` → `\\wsl$\Ubuntu\home\alice`
//! - `/mnt/c/Users` → `C:\Users`
//!
//! # Unicode and Spaces
//!
//! All translations preserve Unicode characters and handle spaces correctly.
//! Round-trip translation produces the original path for supported formats.

use serde::{Deserialize, Serialize};

/// Default WSL distribution name used for WSL2 → Windows translation.
pub const DEFAULT_WSL_DISTRO: &str = "Ubuntu";

// ===========================================================================
// Public API
// ===========================================================================

/// Translate a Windows-style path to a WSL2 Linux path.
///
/// # Rules
///
/// - Drive letter paths: `C:\path` → `/mnt/c/path`
/// - Drive letter only: `C:` → `/mnt/c`
/// - UNC WSL paths: `\\wsl$\Distro\path` → `/path`
/// - Forward slashes are normalized to forward slashes
/// - Drive letter is lowercased
///
/// # Examples
///
/// ```
/// use rusty_stack::platform::path_bridge::windows_to_wsl;
///
/// assert_eq!(windows_to_wsl(r"C:\Users\alice\mlstack"), "/mnt/c/Users/alice/mlstack");
/// assert_eq!(windows_to_wsl(r"\\wsl$\Ubuntu\home\alice\.mlstack"), "/home/alice/.mlstack");
/// ```
pub fn windows_to_wsl(path: &str) -> String {
    let path = path.trim();

    // Handle UNC WSL paths: \\wsl$\Distro\path → /path
    if let Some(rest) = try_strip_unc_wsl_prefix(path) {
        // Normalize backslashes to forward slashes and ensure leading /
        let posix = normalize_to_posix(&rest);
        if posix.is_empty() {
            return "/".to_string();
        }
        if posix.starts_with('/') {
            return posix;
        }
        return format!("/{}", posix);
    }

    // Handle drive letter paths: C:\path → /mnt/c/path
    if let Some(result) = try_drive_letter_to_wsl(path) {
        return result;
    }

    // Fallback: normalize slashes
    normalize_to_posix(path)
}

/// Translate a WSL2 Linux path to a Windows-style path.
///
/// # Rules
///
/// - `/mnt/X/...` → `X:\...` (drive mount paths)
/// - `/home/user` → `\\wsl$\Distro\home\user` (native Linux paths)
///
/// # Examples
///
/// ```
/// use rusty_stack::platform::path_bridge::wsl_to_windows;
///
/// assert_eq!(wsl_to_windows("/mnt/c/Users/alice"), r"C:\Users\alice");
/// assert_eq!(wsl_to_windows("/home/alice/.mlstack"), r"\\wsl$\Ubuntu\home\alice\.mlstack");
/// ```
pub fn wsl_to_windows(path: &str) -> String {
    wsl_to_windows_with_distro(path, DEFAULT_WSL_DISTRO)
}

/// Translate a WSL2 Linux path to a Windows path with a specified distro name.
///
/// Same as [`wsl_to_windows`] but allows customizing the WSL distribution name.
pub fn wsl_to_windows_with_distro(path: &str, distro: &str) -> String {
    let path = path.trim();

    // Handle /mnt/X/... paths → X:\...
    if let Some(result) = try_mnt_to_windows(path) {
        return result;
    }

    // All other paths → \\wsl$\Distro\path
    format!("\\\\wsl$\\{}{}", distro, normalize_to_windows(path))
}

/// Result of a path translation attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathTranslation {
    /// The translated path.
    pub translated: String,
    /// The direction of translation.
    pub direction: TranslationDirection,
}

/// Direction of path translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranslationDirection {
    /// Windows path → WSL2/Linux path.
    WindowsToWsl,
    /// WSL2/Linux path → Windows path.
    WslToWindows,
}

/// Translate a path auto-detecting direction.
///
/// If the path starts with a drive letter (e.g., `C:\`) or a UNC prefix
/// (`\\wsl$`), it is treated as a Windows path. Otherwise, it is treated
/// as a Linux path.
pub fn translate_path(path: &str) -> PathTranslation {
    let path = path.trim();

    if is_windows_path(path) {
        PathTranslation {
            translated: windows_to_wsl(path),
            direction: TranslationDirection::WindowsToWsl,
        }
    } else {
        PathTranslation {
            translated: wsl_to_windows(path),
            direction: TranslationDirection::WslToWindows,
        }
    }
}

/// Check if a path looks like a Windows path.
///
/// Returns true for:
/// - Drive letter paths: `C:\...`
/// - UNC paths: `\\wsl$\...`, `\\server\share\...`
pub fn is_windows_path(path: &str) -> bool {
    let path = path.trim();

    // Drive letter: C:\ or C:/ (single char + colon + separator)
    if path.len() >= 3 {
        let bytes = path.as_bytes();
        if bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'\\' || bytes[2] == b'/')
        {
            return true;
        }
    }

    // UNC: \\ prefix
    path.starts_with(r"\\") || path.starts_with("//")
}

// ===========================================================================
// Private Helpers
// ===========================================================================

/// Try to strip a UNC WSL prefix from a path.
///
/// Handles: `\\wsl$\Distro\path`, `\\wsl.localhost\Distro\path`
/// Returns the remaining path after the distro name if matched.
fn try_strip_unc_wsl_prefix(path: &str) -> Option<String> {
    // Normalize forward slashes to backslashes for matching
    let normalized = path.replace('/', r"\");

    // Try \\wsl$\Distro\ prefix
    if let Some(rest) = normalized.to_lowercase().strip_prefix(r"\\wsl$\") {
        // Find the next backslash (end of distro name)
        if let Some(slash_pos) = rest.find('\\') {
            let distro = &rest[..slash_pos];
            let remaining = &normalized[r"\\wsl$\".len() + distro.len()..];
            return Some(remaining.to_string());
        }
        // Path is just \\wsl$\Distro (no trailing path)
        return Some(String::new());
    }

    // Try \\wsl.localhost\Distro\ prefix
    if let Some(rest) = normalized.to_lowercase().strip_prefix(r"\\wsl.localhost\") {
        if let Some(slash_pos) = rest.find('\\') {
            let distro = &rest[..slash_pos];
            let remaining = &normalized[r"\\wsl.localhost\".len() + distro.len()..];
            return Some(remaining.to_string());
        }
        return Some(String::new());
    }

    None
}

/// Try to convert a drive-letter Windows path to WSL format.
///
/// `C:\path` → `/mnt/c/path`
/// `C:` → `/mnt/c`
fn try_drive_letter_to_wsl(path: &str) -> Option<String> {
    let path = path.trim();

    if path.len() < 2 {
        return None;
    }

    let bytes = path.as_bytes();
    if !bytes[0].is_ascii_alphabetic() || bytes[1] != b':' {
        return None;
    }

    let drive = bytes[0].to_ascii_lowercase();

    if path.len() == 2 {
        // Just "C:" → "/mnt/c"
        return Some(format!("/mnt/{}", drive as char));
    }

    let rest = &path[2..]; // After "C:"
    let rest = normalize_to_posix(rest);

    Some(format!("/mnt/{}{}", drive as char, rest))
}

/// Try to convert a /mnt/X/... path to a Windows drive-letter path.
///
/// `/mnt/c/Users` → `C:\Users`
fn try_mnt_to_windows(path: &str) -> Option<String> {
    let path = path.trim();

    if !path.starts_with("/mnt/") {
        return None;
    }

    let rest = &path[5..]; // After "/mnt/"

    // Extract drive letter (first character after /mnt/)
    let drive = rest.chars().next()?;
    if !drive.is_ascii_alphabetic() {
        return None;
    }

    let after_drive = &rest[drive.len_utf8()..];

    // Drive letter must be followed by / or end of string
    if after_drive.is_empty() {
        return Some(format!("{}:\\", drive.to_ascii_uppercase()));
    }

    if !after_drive.starts_with('/') {
        return None;
    }

    let windows_rest = normalize_to_windows(after_drive);
    Some(format!("{}:{}", drive.to_ascii_uppercase(), windows_rest))
}

/// Normalize path separators to POSIX (forward slash).
fn normalize_to_posix(path: &str) -> String {
    let result = path.replace('\\', "/");

    // Collapse multiple consecutive slashes
    let mut collapsed = String::with_capacity(result.len());
    let mut last_was_slash = false;
    for ch in result.chars() {
        if ch == '/' {
            if !last_was_slash {
                collapsed.push('/');
            }
            last_was_slash = true;
        } else {
            collapsed.push(ch);
            last_was_slash = false;
        }
    }

    collapsed
}

/// Normalize path separators to Windows (backslash).
fn normalize_to_windows(path: &str) -> String {
    path.replace('/', r"\")
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ================================================================
    // VAL-WIN-012: Windows → WSL2 Translation
    // ================================================================

    #[test]
    fn test_windows_to_wsl_basic_drive() {
        assert_eq!(
            windows_to_wsl(r"C:\Users\alice\mlstack"),
            "/mnt/c/Users/alice/mlstack"
        );
    }

    #[test]
    fn test_windows_to_wsl_drive_d() {
        assert_eq!(windows_to_wsl(r"D:\data\file.txt"), "/mnt/d/data/file.txt");
    }

    #[test]
    fn test_windows_to_wsl_root_drive() {
        assert_eq!(windows_to_wsl(r"C:\"), "/mnt/c/");
    }

    #[test]
    fn test_windows_to_wsl_drive_only() {
        assert_eq!(windows_to_wsl(r"C:"), "/mnt/c");
    }

    #[test]
    fn test_windows_to_wsl_forward_slashes() {
        // Windows paths sometimes use forward slashes
        assert_eq!(
            windows_to_wsl("C:/Users/alice/mlstack"),
            "/mnt/c/Users/alice/mlstack"
        );
    }

    #[test]
    fn test_windows_to_wsl_unc_wsl_path() {
        assert_eq!(
            windows_to_wsl(r"\\wsl$\Ubuntu\home\alice\.mlstack"),
            "/home/alice/.mlstack"
        );
    }

    #[test]
    fn test_windows_to_wsl_unc_wsl_path_case_insensitive() {
        // UNC prefix is case-insensitive
        let result = windows_to_wsl(r"\\WSL$\ubuntu\home\test");
        assert!(result.contains("home"), "Should extract path after distro");
    }

    #[test]
    fn test_windows_to_wsl_unc_wsl_localhost() {
        let result = windows_to_wsl(r"\\wsl.localhost\Ubuntu\home\alice");
        assert!(
            result.contains("home"),
            "Should handle wsl.localhost UNC prefix"
        );
    }

    #[test]
    fn test_windows_to_wsl_unc_wsl_distro_only() {
        // Path is just \\wsl$\Distro with no trailing path → root
        let result = windows_to_wsl(r"\\wsl$\Ubuntu");
        assert_eq!(result, "/");
    }

    // ================================================================
    // VAL-WIN-013: WSL2 → Windows Translation
    // ================================================================

    #[test]
    fn test_wsl_to_windows_mnt_path() {
        assert_eq!(wsl_to_windows("/mnt/c/Users/alice"), r"C:\Users\alice");
    }

    #[test]
    fn test_wsl_to_windows_mnt_d() {
        assert_eq!(wsl_to_windows("/mnt/d/data/file.txt"), r"D:\data\file.txt");
    }

    #[test]
    fn test_wsl_to_windows_home_path() {
        assert_eq!(
            wsl_to_windows("/home/alice/.mlstack"),
            r"\\wsl$\Ubuntu\home\alice\.mlstack"
        );
    }

    #[test]
    fn test_wsl_to_windows_custom_distro() {
        assert_eq!(
            wsl_to_windows_with_distro("/home/alice/.mlstack", "Debian"),
            r"\\wsl$\Debian\home\alice\.mlstack"
        );
    }

    #[test]
    fn test_wsl_to_windows_mnt_root() {
        assert_eq!(wsl_to_windows("/mnt/c/"), r"C:\");
    }

    #[test]
    fn test_wsl_to_windows_mnt_drive_only() {
        assert_eq!(wsl_to_windows("/mnt/c"), r"C:\");
    }

    // ================================================================
    // VAL-WIN-014: Unicode and Spaces
    // ================================================================

    #[test]
    fn test_windows_to_wsl_spaces() {
        assert_eq!(
            windows_to_wsl(r"C:\Program Files\ML Stack"),
            "/mnt/c/Program Files/ML Stack"
        );
    }

    #[test]
    fn test_wsl_to_windows_spaces() {
        assert_eq!(
            wsl_to_windows("/mnt/c/Program Files/ML Stack"),
            r"C:\Program Files\ML Stack"
        );
    }

    #[test]
    fn test_windows_to_wsl_unicode() {
        assert_eq!(
            windows_to_wsl(r"C:\Users\日本語\ドキュメント"),
            "/mnt/c/Users/日本語/ドキュメント"
        );
    }

    #[test]
    fn test_wsl_to_windows_unicode() {
        assert_eq!(
            wsl_to_windows("/mnt/c/Users/日本語/ドキュメント"),
            r"C:\Users\日本語\ドキュメント"
        );
    }

    #[test]
    fn test_unicode_roundtrip_windows_to_wsl_and_back() {
        let original = r"C:\Users\alice\文档\ml-stack";
        let wsl = windows_to_wsl(original);
        let back = wsl_to_windows(&wsl);
        assert_eq!(original, &back);
    }

    #[test]
    fn test_spaces_roundtrip_windows_to_wsl_and_back() {
        let original = r"C:\Users\My User\ML Stack Data";
        let wsl = windows_to_wsl(original);
        let back = wsl_to_windows(&wsl);
        assert_eq!(original, &back);
    }

    #[test]
    fn test_unicode_home_to_windows() {
        let path = "/home/alice/文档/数据";
        let win = wsl_to_windows(path);
        assert_eq!(win, r"\\wsl$\Ubuntu\home\alice\文档\数据");
    }

    #[test]
    fn test_unicode_home_roundtrip() {
        let original = "/home/alice/文档/数据";
        let win = wsl_to_windows(original);
        assert_eq!(win, r"\\wsl$\Ubuntu\home\alice\文档\数据");
        let back = windows_to_wsl(&win);
        // UNC path → extracts path after distro, normalizes to POSIX
        assert_eq!(back, "/home/alice/文档/数据");
    }

    // ================================================================
    // Path Detection
    // ================================================================

    #[test]
    fn test_is_windows_path_drive_letter() {
        assert!(is_windows_path(r"C:\Users"));
        assert!(is_windows_path("D:/data"));
    }

    #[test]
    fn test_is_windows_path_unc() {
        assert!(is_windows_path(r"\\wsl$\Ubuntu\home"));
        assert!(is_windows_path("//server/share"));
    }

    #[test]
    fn test_is_not_windows_path() {
        assert!(!is_windows_path("/home/alice"));
        assert!(!is_windows_path("/mnt/c/Users"));
        assert!(!is_windows_path("relative/path"));
    }

    // ================================================================
    // Auto-detection Translation
    // ================================================================

    #[test]
    fn test_translate_path_auto_detect_windows() {
        let result = translate_path(r"C:\Users\alice");
        assert_eq!(result.direction, TranslationDirection::WindowsToWsl);
        assert_eq!(result.translated, "/mnt/c/Users/alice");
    }

    #[test]
    fn test_translate_path_auto_detect_linux() {
        let result = translate_path("/home/alice");
        assert_eq!(result.direction, TranslationDirection::WslToWindows);
        assert!(result.translated.contains("alice"));
    }

    // ================================================================
    // Serde Roundtrips
    // ================================================================

    #[test]
    fn test_path_translation_serde_roundtrip() {
        let t = PathTranslation {
            translated: "/mnt/c/Users".to_string(),
            direction: TranslationDirection::WindowsToWsl,
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: PathTranslation = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn test_translation_direction_serde_roundtrip() {
        for dir in [
            TranslationDirection::WindowsToWsl,
            TranslationDirection::WslToWindows,
        ] {
            let json = serde_json::to_string(&dir).unwrap();
            let back: TranslationDirection = serde_json::from_str(&json).unwrap();
            assert_eq!(dir, back);
        }
    }

    // ================================================================
    // Edge Cases
    // ================================================================

    #[test]
    fn test_empty_path() {
        assert_eq!(windows_to_wsl(""), "");
        assert_eq!(wsl_to_windows(""), r"\\wsl$\Ubuntu");
    }

    #[test]
    fn test_whitespace_trimming() {
        assert_eq!(windows_to_wsl(r"  C:\Users\alice  "), "/mnt/c/Users/alice");
        assert_eq!(
            wsl_to_windows("  /home/alice  "),
            r"\\wsl$\Ubuntu\home\alice"
        );
    }

    #[test]
    fn test_multiple_drives() {
        assert_eq!(windows_to_wsl(r"E:\test"), "/mnt/e/test");
        assert_eq!(windows_to_wsl(r"Z:\foo\bar"), "/mnt/z/foo/bar");
    }

    #[test]
    fn test_mnt_uppercase_drive() {
        // /mnt/C should also work (case-insensitive drive)
        // Our implementation lowercases for mnt matching
        let result = wsl_to_windows("/mnt/C/Users");
        assert_eq!(result, r"C:\Users");
    }

    #[test]
    fn test_deeply_nested_path() {
        let path = r"C:\Users\alice\Documents\Projects\ML Stack\models\v1\checkpoint.bin";
        let wsl = windows_to_wsl(path);
        assert_eq!(
            wsl,
            "/mnt/c/Users/alice/Documents/Projects/ML Stack/models/v1/checkpoint.bin"
        );
        let back = wsl_to_windows(&wsl);
        assert_eq!(path, &back);
    }

    #[test]
    fn test_trailing_slash_handling() {
        assert_eq!(windows_to_wsl(r"C:\Users\"), "/mnt/c/Users/");
        assert_eq!(wsl_to_windows("/mnt/c/"), r"C:\");
    }
}
