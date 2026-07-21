//! `SUDO_ASKPASS` helper for non-interactive privileged operations.
//!
//! `sudo -n` (non-interactive) fails whenever a password is required and no
//! NOPASSWD policy / cached timestamp covers it — which is the common case on a
//! normal desktop. That breaks every privileged step (`/opt/rocm` removal,
//! package-manager purge, and — on Arch — `yay`'s internal `sudo pacman`).
//!
//! This module writes the caller-supplied password to a mode-`0600` file owned
//! by the current user, next to a mode-`0700` askpass script that `cat`s it,
//! inside a private temp directory. Assign the script path to `SUDO_ASKPASS`
//! and run `sudo -A …` (or, for `yay`, `yay --sudo-flags=-A …`): sudo reads the
//! password from the helper with no TTY and no interactive prompt.
//!
//! Notes:
//! - `sudo` always honors `SUDO_ASKPASS` even under `env_reset`, and `-A`
//!   selects the askpass path over a terminal prompt.
//! - The password is never placed in a shell command line (no quoting risk);
//!   it lives only in a root-unreadable, user-owned file for the operation's
//!   duration. [`Askpass`] is an RAII guard that wipes both files on drop.
//! - `yay` must run as the **user**, never under `sudo` (it refuses root for
//!   AUR builds); its internal `sudo pacman` is what consumes the askpass
//!   helper via `--sudo-flags=-A`.

use std::fs::{self, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use tempfile::TempDir;

/// RAII askpass helper.
///
/// Holds a private temp directory (removed on drop) containing the password
/// file (`pw`, `0600`) and the askpass script (`askpass.sh`, `0700`). Use
/// [`Askpass::path`] as the value for `SUDO_ASKPASS`.
pub struct Askpass {
    // Kept alive so the dir (and its files) exist until the guard drops.
    _dir: TempDir,
    script_path: PathBuf,
}

impl Askpass {
    /// Create an askpass helper for `password`.
    ///
    /// Returns a guard whose [`path`](Self::path) is the script to assign to
    /// `SUDO_ASKPASS`. Both the password file and the script are wiped when
    /// the guard is dropped.
    pub fn new(password: &str) -> std::io::Result<Self> {
        // Private dir (0700); tempfile's Builder already uses restrictive perms,
        // set explicitly for clarity.
        let mut builder = tempfile::Builder::new();
        builder.prefix("mlstack-askpass-");
        #[cfg(unix)]
        builder.permissions(fs::Permissions::from_mode(0o700));
        let dir = builder.tempdir()?;

        // Password file — raw bytes, mode 0600. Written verbatim (no newline
        // synthesis) so any password content is handled safely.
        let pw_path = dir.path().join("pw");
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&pw_path)?;
            f.write_all(password.as_bytes())?;
            #[cfg(unix)]
            f.set_permissions(fs::Permissions::from_mode(0o600))?;
        }

        // Askpass script — `cat`s the sibling password file. The password
        // never appears in the script and the path is quoted.
        let script_path = dir.path().join("askpass.sh");
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&script_path)?;
            writeln!(f, "#!/bin/sh")?;
            writeln!(f, "pw_dir=$(dirname \"$0\")")?;
            writeln!(f, "exec cat \"$pw_dir/pw\"")?;
            #[cfg(unix)]
            f.set_permissions(fs::Permissions::from_mode(0o700))?;
        }

        Ok(Self {
            _dir: dir,
            script_path,
        })
    }

    /// The script path to assign to `SUDO_ASKPASS`.
    pub fn path(&self) -> &str {
        self.script_path
            .to_str()
            .expect("tempdir paths are valid UTF-8")
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn script_cats_password_and_is_secure() {
        let helper = Askpass::new("s3cr!t'pass").unwrap();
        let script = std::fs::read_to_string(helper.path()).unwrap();
        assert!(script.contains("#!/bin/sh"));
        assert!(script.contains("exec cat "));
        // The password itself must NOT be embedded in the script.
        assert!(!script.contains("s3cr"));

        // Permissions: script 0700, sibling pw 0600.
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(helper.path())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700);
        let dir = std::path::Path::new(helper.path()).parent().unwrap();
        let pw_mode = std::fs::metadata(dir.join("pw"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(pw_mode & 0o777, 0o600);
    }

    #[test]
    fn askpass_executes_and_returns_password() {
        // End-to-end: the script actually prints the password to stdout.
        let helper = Askpass::new("hunter2").unwrap();
        let out = std::process::Command::new(helper.path())
            .output()
            .expect("askpass script runs");
        assert!(out.status.success(), "askpass exited non-zero");
        assert_eq!(String::from_utf8_lossy(&out.stdout), "hunter2");
    }
}
