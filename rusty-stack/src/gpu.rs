//! GPU classification — the SINGLE source of truth for distinguishing AMD
//! discrete GPUs (dGPUs) from integrated GPUs (iGPUs).
//!
//! # Tenet (v0.3.0 remediation)
//!
//! "The iGPU is CONSISTENTLY filtered from env vars and is NEVER included —
//! EVER. iGPUs are NOT supported and will ONLY cause issues. Intelligent
//! filtering must avoid including the iGPU from env vars or ANY component,
//! while ensuring dGPUs are never missed."
//!
//! This module exists because the crate previously had **three divergent**
//! iGPU classifiers (`installer.rs::is_igpu_name`, `bootstrap/env_setup.rs::
//! is_integrated_gpu_name`, and inline checks in `hardware.rs` /
//! `platform/linux.rs` / `benchmarks`). They disagreed, and three consumers
//! bypassed filtering entirely (`textgen`/`comfyui` hardcoded `"0,1"`;
//! `migraphx_python` trusted inherited `HIP_VISIBLE_DEVICES`). The leaking
//! iGPU broke the MIGraphX/ONNX worker. Everything now routes through here.
//!
//! # Classification rule
//!
//! A device is **integrated** if ANY of:
//! 1. Its PCI device id is in [`INTEGRATED_PCI_DEVICE_IDS`] (structural,
//!    definitive — e.g. Raphael `0x164e`).
//! 2. Its marketing/lspci name matches [`is_integrated_gpu_name`] (rich
//!    curated heuristic: APU codenames, "Ryzen && !RX", APU model suffixes).
//!
//! [`device_is_integrated`] additionally treats a low-VRAM device
//! (`< 4 GiB`) as integrated when the name is ambiguous — but only as a
//! **confirmation**, never as the sole reason to drop a device whose VRAM is
//! unreadable (that would risk missing a real dGPU on kernels without
//! `mem_info_vram_total`). dGPUs (Navi 31/32, 24/16 GB) are never caught.

/// PCI device IDs of known AMD integrated GPUs (APU/iGPU).
///
/// Conservative — only IDs that are *definitively* integrated, so the denylist
/// can never accidentally exclude a discrete card. Extend as new APU families
/// ship.
pub const INTEGRATED_PCI_DEVICE_IDS: &[&str] = &[
    "0x164e", // Raphael — Ryzen 7000 desktop iGPU (e.g. 7800X3D)
    "0x15c8", // Phoenix — Ryzen 7040 mobile APU
    "0x15e0", // Phoenix-Hawk
    "0x1681", // Strix Point (Ryzen AI 300)
    "0x1506", // Renoir — Ryzen 4000/5000 APU
    "0x1638", // Rembrandt — Ryzen 6000/7040 APU
    "0x15d8", // Raven Ridge — Ryzen 2000/3000 APU
    "0x1640", // Cezanne/Barcelo — Ryzen 5000 APU
];

/// VRAM threshold below which a device is treated as integrated (4 GiB).
pub const DISCRETE_MIN_VRAM_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// AMD gfx architectures that are **unambiguously integrated** (iGPU-only).
///
/// These are the structural fallback when a device's marketing name is empty
/// (rocminfo/lspci sometimes report only `Name: gfxNNNN` for the iGPU agent).
/// Every entry is an iGPU-only architecture — no discrete card ships with one —
/// so this list can never cause a dGPU to be missed. Extend cautiously: only
/// add a gfx id once confirmed iGPU-exclusive.
pub const INTEGRATED_GFX_ARCHS: &[&str] = &[
    "gfx1036", // Raphael — Ryzen 7000 desktop iGPU (e.g. 7800X3D)
    "gfx1103", // Phoenix — Ryzen 7040 mobile APU iGPU
];

/// Is this gfx architecture string an iGPU-only architecture?
pub fn is_integrated_by_gfx_arch(gfx: &str) -> bool {
    let gfx = gfx.trim().to_ascii_lowercase();
    INTEGRATED_GFX_ARCHS
        .iter()
        .any(|a| a.eq_ignore_ascii_case(&gfx))
}

/// Curated marketing/lspci name fragments indicating an AMD integrated GPU.
pub const INTEGRATED_NAME_PATTERNS: &[&str] = &[
    // APU codenames (internal names from rocminfo / lspci)
    "Cezanne",
    "Rembrandt",
    "Phoenix",
    "Raphael",
    "Barcelo",
    "Pink Sardine",
    "Yellow Carp",
    "Green Sardine",
    "Hawk Point",
    "Dragon Range",
    "Fire Range",
    "Stella",
    "Krackan",
    "Strix Point",
    "Mendocino",
    "Raven",
    "Picasso",
    "Renoir",
    // Marketing patterns
    "Integrated",
    "iGPU",
    "APU",
    "AMD Radeon Graphics",
    "Radeon Graphics",
    // APU model suffixes
    " 5600G",
    " 5700G",
    " 5600GE",
    " 5700GE",
    " 5700G",
    " 5600GT",
    " 5700GT",
    " 8600G",
    " 8700G",
    " 8500G",
    " 8300G",
    // X3D APUs with integrated graphics (Raphael/Phoenix-based)
    "7800X3D",
    "7950X3D",
    "7900X3D",
    "7700X3D",
    "7600X3D",
    // Mobile APU series
    "Ryzen 7 7735",
    "Ryzen 7 7840",
    "Ryzen 7 8840",
    "Ryzen 5 7535",
    "Ryzen 5 7640",
    "Ryzen 5 8640",
    "Ryzen 9 7945",
    "Ryzen 9 8945",
];

/// Pre-lowercased snapshot of [`INTEGRATED_NAME_PATTERNS`], computed once.
///
/// Avoids re-allocating `pattern.to_lowercase()` on every call to
/// [`is_integrated_gpu_name`] (the patterns are a `const`, so they cannot be
/// stored pre-lowercased at const-eval time).
fn integrated_name_patterns_lower() -> &'static [String] {
    use std::sync::OnceLock;
    static LOWER: OnceLock<Vec<String>> = OnceLock::new();
    LOWER.get_or_init(|| {
        INTEGRATED_NAME_PATTERNS
            .iter()
            .map(|p| p.to_lowercase())
            .collect()
    })
}

/// Does a marketing/lspci/rocminfo name indicate an AMD **integrated** GPU?
///
/// This is the canonical name classifier — the single replacement for the
/// three former divergent copies. Returns `false` for discrete cards
/// ("Radeon RX 7900 XTX", "RX 7800 XT", …) and `true` for APUs/iGPUs.
pub fn is_integrated_gpu_name(marketing_name: &str) -> bool {
    let name_lower = marketing_name.to_lowercase();
    let name_upper = marketing_name.to_uppercase();

    // Case-insensitive match — rocminfo/lspci casing varies ("raphael",
    // "Raphael", "RAPHAEL"). Patterns are compared against the lowercased name
    // via a pre-lowercased snapshot (computed once, not per iteration).
    for pattern in integrated_name_patterns_lower() {
        if name_lower.contains(pattern) {
            return true;
        }
    }

    // "Ryzen" without "RX" ⇒ APU/iGPU. Discrete cards never contain "Ryzen".
    if name_upper.contains("RYZEN") && !name_upper.contains("RX") {
        return true;
    }

    // APU model suffixes: "Ryzen 5 5600G", "Ryzen 7 8700G", "Ryzen 5 5600GE",
    // "Ryzen 5 5600GT" (but not RX/Radeon/XT/XTX). Iterate ALL words after
    // "RYZEN" — standard AMD branding inserts a generation/tier number ("5"/"7")
    // between "Ryzen" and the model, so the suffix is NOT the first word.
    if let Some(pos) = name_upper.find("RYZEN") {
        let after_ryzen = &name_upper[pos + 5..];
        for word in after_ryzen.split_whitespace() {
            if (word.ends_with('G') || word.ends_with("GE") || word.ends_with("GT"))
                && word.len() >= 4
                && !word.starts_with("RX")
                && !word.starts_with("RADEON")
                && !word.contains("XT")
                && !word.contains("XTX")
            {
                return true;
            }
        }
    }

    // Generic "Radeon Graphics" without a model ⇒ iGPU.
    if name_upper.contains("RADEON GRAPHICS")
        && !name_upper.contains("RX")
        && (!marketing_name.chars().any(|c| c.is_ascii_digit())
            || marketing_name.contains("AMD Radeon Graphics"))
    {
        return true;
    }

    false
}

/// Is this PCI device id a known integrated GPU?
///
/// `pci_device_id` may be lowercase/uppercase and with or without a `0x`
/// prefix (e.g. `"0x164e"`, `"164E"`).
pub fn is_integrated_by_pci_id(pci_device_id: &str) -> bool {
    let normalized = pci_device_id.trim().to_ascii_lowercase();
    let normalized = normalized.trim_start_matches("0x");
    INTEGRATED_PCI_DEVICE_IDS
        .iter()
        .any(|id| id.trim_start_matches("0x").eq_ignore_ascii_case(normalized))
}

/// Combined, fail-safe classification of a single device.
///
/// Inputs are all optional — provide whatever the caller has. Returns `true`
/// (integrated) when ANY of:
/// - the PCI device id is a known iGPU (`pci_device_id`), OR
/// - the gfx architecture is iGPU-only (`gfx_arch`, e.g. `gfx1036` Raphael) —
///   the structural fallback when the marketing name is empty, OR
/// - the name classifies as integrated (`name`), OR
/// - the name is ambiguous AND the device's VRAM is readable and below
///   [`DISCRETE_MIN_VRAM_BYTES`] (confirmation only).
///
/// Unreadable VRAM alone NEVER classifies a device as integrated (that would
/// risk dropping a real dGPU on kernels lacking `mem_info_vram_total`); the
/// PCI id, gfx arch, and name are the authoritative signals. A dGPU is never
/// missed: every exclusion requires a positive iGPU signal.
pub fn device_is_integrated(
    name: Option<&str>,
    pci_device_id: Option<&str>,
    gfx_arch: Option<&str>,
    vram_bytes: Option<u64>,
) -> bool {
    if let Some(id) = pci_device_id {
        if is_integrated_by_pci_id(id) {
            return true;
        }
    }
    if let Some(g) = gfx_arch {
        if !g.trim().is_empty() && is_integrated_by_gfx_arch(g) {
            return true;
        }
    }
    if let Some(name) = name {
        if !name.trim().is_empty() && is_integrated_gpu_name(name) {
            return true;
        }
        // VRAM confirmation: only when the name is ambiguous (carries NO
        // positive dGPU signal) AND VRAM is concretely readable + low. Never
        // the sole reason, and never VRAM-rejects an explicitly-named dGPU
        // (e.g. a <4 GiB Radeon RX / Radeon Pro / Instinct / FirePro card) —
        // "dGPUs are never missed."
        //
        // The marker is an EXPLICIT dGPU qualifier only ("Radeon RX", " RX ",
        // "Radeon Pro", "Instinct", "FirePro"). A bare brand name ("Radeon",
        // "AMD Radeon") with no model qualifier is ambiguous — that is the APU
        // reporting style — and must fall through to the VRAM gate rather than
        // being treated as authoritative discrete. (Matching on "RADEON" alone
        // would shield bare iGPU marketing names from the gate, leaking the
        // iGPU.) RDNA iGPUs with model numbers ("Radeon 780M", "Radeon 680M")
        // carry no qualifier either, so they too fall through and are gated by
        // VRAM — correct, since they are APUs. The only cards lost are
        // pre-RDNA "Radeon HD" dGPUs, which no ROCm release supports.
        if let Some(vram) = vram_bytes {
            let upper = name.to_ascii_uppercase();
            let has_discrete_marker = upper.contains("RADEON RX")
                || upper.contains(" RX ")
                || upper.contains("RADEON PRO")
                || upper.contains("INSTINCT")
                || upper.contains("FIREPRO");
            if !name.trim().is_empty() && !has_discrete_marker && vram < DISCRETE_MIN_VRAM_BYTES {
                return true;
            }
        }
    }
    false
}

/// Read a PCI device's `mem_info_vram_total` (bytes) from sysfs, if available.
///
/// `pci_slot` is the `BB:DD.F` form (with or without a `0000:` domain prefix).
pub fn read_vram_bytes(pci_slot: &str) -> Option<u64> {
    let slot = pci_slot.trim().trim_start_matches("0000:");
    let path = format!("/sys/bus/pci/devices/0000:{slot}/mem_info_vram_total");
    let value = std::fs::read_to_string(&path).ok()?;
    value.trim().parse::<u64>().ok()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_cards_are_not_integrated() {
        assert!(!is_integrated_gpu_name("AMD Radeon RX 7900 XTX"));
        assert!(!is_integrated_gpu_name("Radeon RX 7800 XT"));
        assert!(!is_integrated_gpu_name("AMD Radeon RX 6900 XT"));
        assert!(!is_integrated_gpu_name("AMD Radeon RX 9070 XT"));
    }

    #[test]
    fn apu_codenames_are_integrated() {
        assert!(is_integrated_gpu_name("Raphael"));
        assert!(is_integrated_gpu_name("raphael")); // case-insensitive (rocminfo/lspci varies)
        assert!(is_integrated_gpu_name("RAPHAEL"));
        assert!(!is_integrated_gpu_name("gfx1036")); // raw gfx arch is NOT a name signal
        assert!(is_integrated_gpu_name("AMD Cezanne (Radeon Graphics)"));
        assert!(is_integrated_gpu_name("Phoenix"));
        assert!(is_integrated_gpu_name("Rembrandt"));
        assert!(is_integrated_gpu_name("Strix Point")); // Ryzen AI 300 APU
    }

    #[test]
    fn asus_strix_dgpu_is_not_integrated() {
        // "Strix Point" is an APU codename, but "ROG Strix" is an Asus dGPU
        // board line — must NOT be filtered. (Regression guard.)
        assert!(!is_integrated_gpu_name("ASUS ROG Strix Radeon RX 7900 XTX"));
        assert!(!is_integrated_gpu_name("ROG Strix LC Radeon RX 7800 XT"));
    }

    #[test]
    fn ryzen_cpus_with_igpu_are_integrated() {
        assert!(is_integrated_gpu_name(
            "AMD Ryzen 7 7800X3D 8-Core Processor"
        ));
        assert!(is_integrated_gpu_name(
            "AMD Ryzen 9 7950X3D 16-Core Processor"
        ));
        assert!(is_integrated_gpu_name("AMD Ryzen 5 8600G 6-Core Processor"));
        assert!(is_integrated_gpu_name("AMD Ryzen 7 8700G"));
    }

    #[test]
    fn ryzen_apu_suffix_without_amd_prefix_is_integrated() {
        // Regression: the old " RYZEN " search + first-word check never matched
        // "Ryzen 5 5600G" (no leading space, model number is the 2nd word).
        assert!(is_integrated_gpu_name("Ryzen 5 5600G"));
        assert!(is_integrated_gpu_name("Ryzen 7 5700G"));
        assert!(is_integrated_gpu_name("Ryzen 5 5600GE"));
        assert!(is_integrated_gpu_name("Ryzen 5 5600GT"));
        assert!(is_integrated_gpu_name("Ryzen 7 8700G"));
    }

    #[test]
    fn low_vram_discrete_cards_are_not_vram_rejected() {
        // Regression: <4 GiB must NOT drop an explicitly-named dGPU
        // ("dGPUs are never missed"). Older Radeon RX/Pro cards <4 GB.
        let low = 2 * 1024 * 1024 * 1024u64;
        assert!(!device_is_integrated(
            Some("AMD Radeon RX 550"),
            None,
            None,
            Some(low)
        ));
        assert!(!device_is_integrated(
            Some("AMD Radeon Pro WX 2100"),
            None,
            None,
            Some(low)
        ));
        assert!(!device_is_integrated(
            Some("AMD Instinct MI50"),
            None,
            None,
            Some(low)
        ));
        // Ambiguous name + low VRAM is still treated as integrated.
        assert!(device_is_integrated(
            Some("AMD Device"),
            Some("0x9999"),
            None,
            Some(512 * 1024 * 1024)
        ));
    }

    #[test]
    fn bare_radeon_marketing_name_is_vram_gated() {
        // Regression (PR #21 re-review): a bare brand string with no model
        // qualifier, gfx arch, or PCI id at <4 GiB is the iGPU reporting style
        // and must be VRAM-gated to integrated — NOT shielded by a "RADEON"
        // substring. The marker list matches only explicit qualifiers.
        let low = 2 * 1024 * 1024 * 1024u64;
        assert!(device_is_integrated(Some("Radeon"), None, None, Some(low)));
        assert!(device_is_integrated(
            Some("AMD Radeon"),
            None,
            None,
            Some(low)
        ));

        // RDNA iGPU models with no qualifier ("Radeon 780M" = Phoenix/Strix APU)
        // also fall through to the VRAM gate and are correctly integrated — a
        // naive "has-digit ⇒ discrete" rule would misclassify these as dGPUs.
        assert!(device_is_integrated(
            Some("AMD Radeon 780M"),
            None,
            None,
            Some(low)
        ));
        assert!(device_is_integrated(
            Some("AMD Radeon 680M"),
            None,
            None,
            Some(low)
        ));

        // Same bare names with ample VRAM are NOT integrated (fail-safe: a
        // bare name is never an authoritative dGPU signal, but ample VRAM
        // declines to reclassify).
        let high = 16 * 1024 * 1024 * 1024u64;
        assert!(!device_is_integrated(
            Some("Radeon"),
            None,
            None,
            Some(high)
        ));

        // Explicitly-named dGPUs survive the gate regardless of VRAM.
        assert!(!device_is_integrated(
            Some("AMD Radeon RX 7900 XTX"),
            None,
            None,
            Some(low)
        ));
    }

    #[test]
    fn generic_radeon_graphics_is_integrated() {
        assert!(is_integrated_gpu_name("AMD Radeon Graphics"));
        assert!(!is_integrated_gpu_name("AMD Radeon RX Graphics 7900"));
    }

    #[test]
    fn pci_id_denylist_catches_raphael() {
        assert!(is_integrated_by_pci_id("0x164e"));
        assert!(is_integrated_by_pci_id("164E"));
        assert!(is_integrated_by_pci_id("0x164E"));
        assert!(!is_integrated_by_pci_id("0x744c")); // Navi 31 dGPU (7900 XTX)
        assert!(!is_integrated_by_pci_id("0x747e")); // Navi 32 dGPU (7800 XT)
    }

    #[test]
    fn device_is_integrated_combines_signals() {
        // Name-only iGPU.
        assert!(device_is_integrated(Some("Raphael"), None, None, None));
        // PCI id-only iGPU.
        assert!(device_is_integrated(None, Some("0x164e"), None, None));
        // gfx-arch-only iGPU (nameless iGPU agent fallback).
        assert!(device_is_integrated(None, None, Some("gfx1036"), None));
        assert!(device_is_integrated(None, None, Some("gfx1103"), None));
        // dGPU by name with large VRAM ⇒ not integrated.
        assert!(!device_is_integrated(
            Some("AMD Radeon RX 7900 XTX"),
            Some("0x744c"),
            Some("gfx1100"),
            Some(24 * 1024 * 1024 * 1024)
        ));
        // Ambiguous name + low readable VRAM ⇒ integrated (confirmation).
        assert!(device_is_integrated(
            Some("AMD Device"),
            Some("0x9999"),
            None,
            Some(512 * 1024 * 1024)
        ));
        // Ambiguous name + UNREADABLE VRAM ⇒ NOT integrated (never miss a dGPU).
        assert!(!device_is_integrated(
            Some("AMD Device"),
            Some("0x9999"),
            None,
            None
        ));
    }

    #[test]
    fn gfx_arch_signal_is_safe_for_dgpus() {
        // iGPU-only archs are caught.
        assert!(is_integrated_by_gfx_arch("gfx1036"));
        assert!(is_integrated_by_gfx_arch("GFX1103"));
        // dGPU archs are NEVER caught (no false exclusions).
        assert!(!is_integrated_by_gfx_arch("gfx1100"));
        assert!(!is_integrated_by_gfx_arch("gfx1101"));
        assert!(!is_integrated_by_gfx_arch("gfx1030"));
    }
}
