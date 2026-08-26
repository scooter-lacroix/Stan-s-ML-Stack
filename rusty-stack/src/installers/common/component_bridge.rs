//! Cross-component recognition glue.
//!
//! Component pairs that interoperate at runtime need to find each other after
//! an install of EITHER side, in ANY order. The first pair: FastVideo ships a
//! ComfyUI custom-node pack as part of its wheel (top-level `comfyui` package
//! exposing NODE_CLASS_MAPPINGS), while stock ComfyUI only scans its
//! `custom_nodes/` directory. A symlink from the clone into the managed env
//! keeps recognition live across every future FastVideo reinstall because it
//! points INTO the managed environment rather than copying content.

use std::io;
use std::path::{Path, PathBuf};

/// Outcome of trying to lay a custom-node link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkOutcome {
    /// Symlink created.
    Created,
    /// Existing symlink pointed elsewhere; re-pointed.
    Refreshed,
    /// Symlink already correct; no action.
    AlreadyLinked,
    /// A real directory/file occupies the slot; left untouched.
    OccupiedByRealDir,
}

/// Ensure `<comfy_install_dir>/custom_nodes/<name>` is a symlink to `node_src`.
///
/// `node_src` should be inside the managed environment so component upgrades
/// propagate without re-wiring.
pub fn ensure_custom_node_link(
    comfy_install_dir: &Path,
    node_src: &Path,
    name: &str,
) -> io::Result<LinkOutcome> {
    let custom_dir = comfy_install_dir.join("custom_nodes");
    std::fs::create_dir_all(&custom_dir)?;
    let link = custom_dir.join(name);

    match std::fs::symlink_metadata(&link) {
        Ok(meta) => {
            if !meta.file_type().is_symlink() {
                return Ok(LinkOutcome::OccupiedByRealDir);
            }
            if std::fs::read_link(&link)?.as_path() == node_src {
                return Ok(LinkOutcome::AlreadyLinked);
            }
            std::fs::remove_file(&link)?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(node_src, &link)?;
            Ok(LinkOutcome::Refreshed)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            #[cfg(unix)]
            std::os::unix::fs::symlink(node_src, &link)?;
            Ok(LinkOutcome::Created)
        }
        Err(err) => Err(err),
    }
}

/// Where the FastVideo-comfy node pack lands: the `comfyui` package inside
/// the managed Python environment (resolvable only through Python since the
/// lib dir layout belongs to the interpreter).
pub fn expected_comfyui_dir(home: &str) -> PathBuf {
    PathBuf::from(format!("{home}/ComfyUI"))
}

#[cfg(test)]
mod tests {
    use super::{ensure_custom_node_link, expected_comfyui_dir, LinkOutcome};
    use std::path::PathBuf;

    fn tempdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "component-bridge-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn links_then_reports_already_linked() {
        let root = tempdir("ok");
        let src = root.join("managed/comfyui");
        std::fs::create_dir_all(&src).unwrap();
        let comfy = root.join("ComfyUI");

        assert_eq!(
            ensure_custom_node_link(&comfy, &src, "fastvideo_nodes").unwrap(),
            LinkOutcome::Created
        );
        assert_eq!(
            ensure_custom_node_link(&comfy, &src, "fastvideo_nodes").unwrap(),
            LinkOutcome::AlreadyLinked
        );
        let link = comfy.join("custom_nodes/fastvideo_nodes");
        assert!(link.is_symlink());
        assert_eq!(std::fs::read_link(&link).unwrap(), src);
    }

    #[test]
    fn refreshes_stale_pointer_and_spares_real_dirs() {
        let root = tempdir("refresh");
        let old = root.join("old-pack");
        let new = root.join("managed/comfyui");
        std::fs::create_dir_all(old.join("x")).unwrap();
        std::fs::create_dir_all(&new).unwrap();
        let comfy = root.join("ComfyUI");

        ensure_custom_node_link(&comfy, &old, "fastvideo_nodes").unwrap();
        assert_eq!(
            ensure_custom_node_link(&comfy, &new, "fastvideo_nodes").unwrap(),
            LinkOutcome::Refreshed
        );
        assert_eq!(
            std::fs::read_link(comfy.join("custom_nodes/fastvideo_nodes")).unwrap(),
            new
        );

        // real directory occupying the slot must never be clobbered
        let occupied = comfy.join("custom_nodes/fastvideo_nodes");
        std::fs::remove_file(&occupied).unwrap();
        std::fs::create_dir_all(&occupied).unwrap();
        assert_eq!(
            ensure_custom_node_link(&comfy, &new, "fastvideo_nodes").unwrap(),
            LinkOutcome::OccupiedByRealDir
        );
    }

    #[test]
    fn expected_comfyui_dir_honors_home() {
        assert_eq!(
            expected_comfyui_dir("/users/op"),
            PathBuf::from("/users/op/ComfyUI")
        );
    }
}
