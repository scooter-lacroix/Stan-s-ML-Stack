//! Installed-component registry — the single source of truth for what Rusty
//! has installed, where, and whether a *core* component is sealed against
//! override.
//!
//! Persisted at `~/.mlstack/installed.json`. This module is deliberately
//! self-contained (load / save / mutate) so it can be threaded through the
//! installer dispatch (see `installer::run_native_installer`) and consulted by:
//! - dep-sourcing / no-override (refuse to reinstall a sealed core component)
//! - verification (is this component already known-good?)
//! - uninstall (enumerate pip packages + locations to remove)
//!
//! # Design notes
//!
//! - IDs are free-form `String`s matching the existing
//!   `component_status::is_component_installed_by_id` identifiers (`rocm`,
//!   `pytorch`, `triton`, `aiter`, …).
//! - `sealed` marks a **core** component (rocm / pytorch / triton) that later
//!   installers must reuse, never override. See Stage 3 of the v0.3.0
//!   remediation (tenet: single-source deps / no-override).
//! - Timestamps are caller-supplied ISO-8601 strings (chrono `Utc::now`) so the
//!   registry stays pure with respect to I/O.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::platform::environment::mlstack_root;

/// A single installed component record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledComponent {
    /// Component identifier (e.g. `rocm`, `pytorch`, `triton`, `aiter`).
    pub id: String,
    /// Installed version, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Source index / wheel index URL used (for pip components), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_index: Option<String>,
    /// Install location (venv prefix, `/opt/rocm`, clone dir, …), if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// `true` for sealed **core** components that must never be overridden.
    #[serde(default)]
    pub sealed: bool,
    /// ISO-8601 install timestamp (caller-supplied).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installed_at: Option<String>,
    /// Free-form pip package names this component owns (for uninstall).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pip_packages: Vec<String>,
}

/// The persisted registry of installed components.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstalledComponentRegistry {
    #[serde(default)]
    components: Vec<InstalledComponent>,
}

impl InstalledComponentRegistry {
    /// Path to the persisted registry file: `~/.mlstack/installed.json`.
    pub fn path() -> PathBuf {
        mlstack_root().join("installed.json")
    }

    /// Load the registry from disk, returning an empty one if absent or
    /// unreadable (never panics — a corrupt registry degrades to empty).
    pub fn load() -> Self {
        let path = Self::path();
        match fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Persist the registry to disk, creating `~/.mlstack/` first.
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Load then save — normalizes the file on disk (used after manual edits).
    pub fn reload_and_save() -> anyhow::Result<()> {
        Self::load().save()
    }

    /// Record (or update) a component. If `sealed` is requested on a component
    /// that is already sealed, the seal is preserved (never accidentally
    /// unsealed via re-mark without explicit [`Self::unseal`]).
    pub fn mark_installed(&mut self, component: InstalledComponent) {
        if let Some(existing) = self.components.iter_mut().find(|c| c.id == component.id) {
            let preserve_seal = existing.sealed && !component.sealed;
            *existing = component;
            if preserve_seal {
                existing.sealed = true;
            }
        } else {
            self.components.push(component);
        }
    }

    /// Convenience: mark a core component as installed + sealed.
    pub fn seal_core(
        &mut self,
        id: &str,
        version: Option<String>,
        source_index: Option<String>,
        location: Option<String>,
        pip_packages: Vec<String>,
        installed_at: Option<String>,
    ) {
        self.mark_installed(InstalledComponent {
            id: id.to_string(),
            version,
            source_index,
            location,
            sealed: true,
            installed_at,
            pip_packages,
        });
    }

    /// Lookup a component by id.
    pub fn get(&self, id: &str) -> Option<&InstalledComponent> {
        self.components.iter().find(|c| c.id == id)
    }

    /// Is a component with this id recorded as installed?
    pub fn is_installed(&self, id: &str) -> bool {
        self.components.iter().any(|c| c.id == id)
    }

    /// Is this a sealed core component?
    pub fn is_sealed(&self, id: &str) -> bool {
        self.get(id).is_some_and(|c| c.sealed)
    }

    /// All recorded component ids.
    pub fn installed_ids(&self) -> HashSet<String> {
        self.components.iter().map(|c| c.id.clone()).collect()
    }

    /// Every pip package name owned by any recorded component (for uninstall).
    pub fn all_pip_packages(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for c in &self.components {
            for p in &c.pip_packages {
                if seen.insert(p.clone()) {
                    out.push(p.clone());
                }
            }
        }
        out
    }

    /// Explicitly unseal a core component (only via deliberate user action,
    /// e.g. `--unseal-core`). Returns `true` if it was sealed and is now not.
    pub fn unseal(&mut self, id: &str) -> bool {
        if let Some(c) = self.components.iter_mut().find(|c| c.id == id) {
            let was = c.sealed;
            c.sealed = false;
            was
        } else {
            false
        }
    }

    /// Remove a component record (does not uninstall — see Stage 6).
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.components.len();
        self.components.retain(|c| c.id != id);
        self.components.len() != before
    }

    /// Clear all records.
    pub fn clear(&mut self) {
        self.components.clear();
    }

    /// Number of recorded components.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Is the registry empty?
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Iterate over recorded components.
    pub fn iter(&self) -> impl Iterator<Item = &InstalledComponent> {
        self.components.iter()
    }
}

/// The set of ids treated as **core** (sealed-on-install) components.
pub const CORE_COMPONENT_IDS: &[&str] = &["rocm", "pytorch", "triton"];

/// Additional **install-once** components the tenet names ("aiter, flash
/// attention, etc.") — expensive ROCm builds that, once installed, must be
/// reused and never overridden. Sealed alongside the cores.
pub const SEALED_INSTALL_ONCE_IDS: &[&str] = &[
    "aiter",
    "flash-attn",
    "flash_attention",
    "flash_attention_amd",
    "rccl",
    "migraphx",
    "bitsandbytes",
];

/// Is the given id a core component?
pub fn is_core_component(id: &str) -> bool {
    CORE_COMPONENT_IDS.contains(&id)
}

/// Should this component be **sealed** on install (core OR install-once)?
///
/// A sealed component is reused (not reinstalled) on subsequent installs and
/// protected from force-reinstall unless `MLSTACK_UNSEAL_CORE=1` — the tenet's
/// "once installed, NEVER overridden" guarantee.
pub fn should_seal_component(id: &str) -> bool {
    is_core_component(id) || SEALED_INSTALL_ONCE_IDS.contains(&id)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn reg(id: &str, sealed: bool) -> InstalledComponent {
        InstalledComponent {
            id: id.to_string(),
            version: Some("1.0".into()),
            source_index: None,
            location: None,
            sealed,
            installed_at: None,
            pip_packages: vec![format!("{id}-pkg")],
        }
    }

    #[test]
    fn mark_and_lookup() {
        let mut r = InstalledComponentRegistry::default();
        r.mark_installed(reg("pytorch", true));
        assert!(r.is_installed("pytorch"));
        assert!(r.is_sealed("pytorch"));
        assert!(!r.is_installed("rocm"));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn seal_is_preserved_on_remark() {
        let mut r = InstalledComponentRegistry::default();
        r.mark_installed(reg("pytorch", true));
        // A later non-sealed remark with NEW non-seal fields must update those
        // fields while preserving the seal (not skip the update entirely).
        let updated = InstalledComponent {
            id: "pytorch".to_string(),
            version: Some("2.9.0".into()),
            source_index: Some("https://download.pytorch.org/whl/rocm7.6".into()),
            location: None,
            sealed: false,
            installed_at: None,
            pip_packages: vec!["torch".into(), "torchvision".into()],
        };
        r.mark_installed(updated);
        assert!(
            r.is_sealed("pytorch"),
            "seal must survive a non-sealed remark"
        );
        let got = r.get("pytorch").expect("component present");
        assert_eq!(got.version.as_deref(), Some("2.9.0"), "version must update");
        assert_eq!(
            got.source_index.as_deref(),
            Some("https://download.pytorch.org/whl/rocm7.6")
        );
        assert_eq!(got.pip_packages, vec!["torch", "torchvision"]);
    }

    #[test]
    fn unseal_is_explicit() {
        let mut r = InstalledComponentRegistry::default();
        r.mark_installed(reg("pytorch", true));
        assert!(r.unseal("pytorch"));
        assert!(!r.is_sealed("pytorch"));
    }

    #[test]
    fn all_pip_packages_dedups() {
        let mut r = InstalledComponentRegistry::default();
        let mut a = reg("pytorch", true);
        a.pip_packages = vec!["torch".into(), "torchvision".into()];
        let mut b = reg("vllm", false);
        b.pip_packages = vec!["vllm".into(), "torch".into()]; // torch dup
        r.mark_installed(a);
        r.mark_installed(b);
        let pkgs = r.all_pip_packages();
        let torchs = pkgs.iter().filter(|p| *p == "torch").count();
        assert_eq!(torchs, 1, "torch must appear once: {pkgs:?}");
        assert!(pkgs.contains(&"vllm".to_string()));
    }

    #[test]
    fn remove_and_clear() {
        let mut r = InstalledComponentRegistry::default();
        r.mark_installed(reg("rocm", true));
        r.mark_installed(reg("pytorch", true));
        assert!(r.remove("rocm"));
        assert!(!r.is_installed("rocm"));
        r.clear();
        assert!(r.is_empty());
    }

    #[test]
    fn load_missing_is_empty() {
        // Point HOME at a temp dir so ~/.mlstack/installed.json is absent.
        let dir = tempfile::tempdir().unwrap();
        let saved_home = std::env::var("MLSTACK_USER_HOME").ok();
        std::env::set_var(
            "MLSTACK_USER_HOME",
            dir.path().to_string_lossy().to_string(),
        );
        let r = InstalledComponentRegistry::load();
        assert!(r.is_empty());
        match saved_home {
            Some(v) => std::env::set_var("MLSTACK_USER_HOME", v),
            None => std::env::remove_var("MLSTACK_USER_HOME"),
        }
    }

    #[test]
    fn save_then_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let saved_home = std::env::var("MLSTACK_USER_HOME").ok();
        std::env::set_var(
            "MLSTACK_USER_HOME",
            dir.path().to_string_lossy().to_string(),
        );

        let mut r = InstalledComponentRegistry::default();
        r.seal_core(
            "pytorch",
            Some("2.4.0".into()),
            Some("https://download.pytorch.org/whl/rocm6.2".into()),
            Some("/home/u/.mlstack/global".into()),
            vec!["torch".into()],
            Some("2026-06-24T00:00:00Z".into()),
        );
        r.save().unwrap();
        assert!(InstalledComponentRegistry::path().exists());

        let loaded = InstalledComponentRegistry::load();
        assert!(loaded.is_sealed("pytorch"));
        assert_eq!(
            loaded.get("pytorch").unwrap().version.as_deref(),
            Some("2.4.0")
        );

        match saved_home {
            Some(v) => std::env::set_var("MLSTACK_USER_HOME", v),
            None => std::env::remove_var("MLSTACK_USER_HOME"),
        }
    }

    #[test]
    fn core_ids_classify_correctly() {
        assert!(is_core_component("rocm"));
        assert!(is_core_component("pytorch"));
        assert!(is_core_component("triton"));
        assert!(!is_core_component("vllm"));
        assert!(!is_core_component("deepspeed"));
    }
}
