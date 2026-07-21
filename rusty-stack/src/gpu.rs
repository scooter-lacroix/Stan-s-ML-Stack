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
// AMD GPU detection from sysfs — PCI class filter, VRAM, gfx arch resolution
// ===========================================================================

/// AMD discrete GPU discovered via sysfs.
///
/// All fields optional except `pci_slot` and `is_integrated`. Detectors
/// populate from `/sys/class/drm/card*/device` and PCI tables.
#[derive(Debug, Clone)]
pub struct AmdGpu {
    /// PCI slot in `BB:DD.F` form (canonicalized, no `0000:` prefix).
    pub pci_slot: String,
    /// PCI device ID (hex string, e.g. `"0x744c"` for Navi 31).
    pub pci_device_id: String,
    /// Marketing name from lspci/rocminfo if available.
    pub marketing_name: Option<String>,
    /// gfx architecture (e.g. `"gfx1100"` for Navi 31).
    pub gfx_arch: Option<String>,
    /// Total VRAM in bytes from `mem_info_vram_total`.
    pub vram_bytes: Option<u64>,
    /// `true` if classified as integrated by [`device_is_integrated`].
    pub is_integrated: bool,
}

/// Authoritative PCI device ID → gfx arch mapping for AMD discrete GPUs.
///
/// Sourced from `/usr/share/hwdata/pci.ids`. ONLY dGPUs — iGPU IDs handled
/// separately by [`INTEGRATED_PCI_DEVICE_IDS`].
pub const DISCRETE_PCI_ID_TO_GFX: &[(&str, &str)] = &[
    // RDNA4
    ("0x7550", "gfx1201"),
    ("0x7551", "gfx1201"),
    ("0x7590", "gfx1200"),
    // RDNA3 Navi31
    ("0x744c", "gfx1100"),
    ("0x7448", "gfx1100"),
    ("0x7449", "gfx1100"),
    ("0x744a", "gfx1100"),
    ("0x744b", "gfx1100"),
    ("0x745e", "gfx1100"),
    // RDNA3 Navi32
    ("0x747e", "gfx1101"),
    ("0x7470", "gfx1101"),
    ("0x7460", "gfx1101"),
    ("0x7461", "gfx1101"),
    // RDNA3 Navi33
    ("0x7480", "gfx1102"),
    ("0x7483", "gfx1102"),
    ("0x7489", "gfx1102"),
    ("0x749f", "gfx1102"),
    ("0x73f0", "gfx1102"),
    // RDNA2 Navi21
    ("0x73bf", "gfx1030"),
    ("0x73af", "gfx1030"),
    ("0x73a5", "gfx1030"),
    ("0x73a1", "gfx1030"),
    ("0x73a2", "gfx1030"),
    ("0x73a3", "gfx1030"),
    // RDNA2 Navi22
    ("0x73df", "gfx1031"),
    ("0x73c3", "gfx1031"),
    // RDNA2 Navi23
    ("0x73ff", "gfx1032"),
    ("0x73ef", "gfx1032"),
    ("0x73e0", "gfx1032"),
    ("0x73e1", "gfx1032"),
    ("0x73e3", "gfx1032"),
    // RDNA2 Navi24
    ("0x743f", "gfx1034"),
    ("0x7424", "gfx1034"),
    ("0x7421", "gfx1034"),
    ("0x7422", "gfx1034"),
    ("0x7423", "gfx1034"),
];

/// Normalize PCI device ID (lowercase, strip `0x`).
fn normalize_pci_id(id: &str) -> String {
    id.trim()
        .to_ascii_lowercase()
        .trim_start_matches("0x")
        .to_string()
}

/// Look up gfx arch by PCI device ID (normalized, case-insensitive).
pub fn pci_id_to_gfx(pci_device_id: &str) -> Option<&'static str> {
    let normalized = normalize_pci_id(pci_device_id);
    DISCRETE_PCI_ID_TO_GFX
        .iter()
        .find(|(id, _)| normalize_pci_id(id) == normalized)
        .map(|(_, gfx)| *gfx)
}

/// Resolve gfx arch from marketing name (canonical fallback).
///
/// Copies logic from `platform/linux.rs::get_correct_gfx_from_marketing_name`
/// and extends with generic "Navi NN" patterns. Returns `None` if unknown.
pub fn gfx_from_marketing_name(name: &str) -> Option<&'static str> {
    let name_lower = name.to_lowercase();

    // Generic "Navi NN" patterns
    if name_lower.contains("navi 48") || name_lower.contains("navi48") {
        return Some("gfx1201");
    }
    if name_lower.contains("navi 44") || name_lower.contains("navi44") {
        return Some("gfx1200");
    }
    if name_lower.contains("navi 31") || name_lower.contains("navi31") {
        return Some("gfx1100");
    }
    if name_lower.contains("navi 32") || name_lower.contains("navi32") {
        return Some("gfx1101");
    }
    if name_lower.contains("navi 33") || name_lower.contains("navi33") {
        return Some("gfx1102");
    }
    if name_lower.contains("navi 21") || name_lower.contains("navi21") {
        return Some("gfx1030");
    }
    if name_lower.contains("navi 22") || name_lower.contains("navi22") {
        return Some("gfx1031");
    }
    if name_lower.contains("navi 23") || name_lower.contains("navi23") {
        return Some("gfx1032");
    }
    if name_lower.contains("navi 24") || name_lower.contains("navi24") {
        return Some("gfx1034");
    }

    // RDNA 3 (Navi 3x) — gfx1100/gfx1101/gfx1102
    if name_lower.contains("7900 xtx") || name_lower.contains("7900xtx") {
        return Some("gfx1100");
    }
    if name_lower.contains("7900 gre") || name_lower.contains("7900gre") {
        return Some("gfx1100");
    }
    if name_lower.contains("7900 xt") || name_lower.contains("7900xt") {
        return Some("gfx1100");
    }
    if name_lower.contains("7800 xt") || name_lower.contains("7800xt") {
        return Some("gfx1101");
    }
    if name_lower.contains("7800 gre") || name_lower.contains("7800gre") {
        return Some("gfx1101");
    }
    if name_lower.contains("7700 xt") || name_lower.contains("7700xt") {
        return Some("gfx1101");
    }
    if name_lower.contains("7600 xt")
        || name_lower.contains("7600xt")
        || name_lower.contains("7600")
    {
        return Some("gfx1102");
    }

    // RDNA 4 (Navi 4x) — gfx1200/gfx1201
    if name_lower.contains("9070 xt")
        || name_lower.contains("9070xt")
        || name_lower.contains("9070 gre")
        || name_lower.contains("9070gre")
    {
        return Some("gfx1201");
    }
    if name_lower.contains("9060") {
        return Some("gfx1200");
    }

    // RDNA 2 (Navi 2x) — gfx1030/gfx1031/gfx1032/gfx1034
    if name_lower.contains("6950 xt")
        || name_lower.contains("6950xt")
        || name_lower.contains("6900 xt")
        || name_lower.contains("6900xt")
    {
        return Some("gfx1030");
    }
    if name_lower.contains("6800 xt")
        || name_lower.contains("6800xt")
        || name_lower.contains("6800")
        || name_lower.contains("6900")
    {
        return Some("gfx1030");
    }
    if name_lower.contains("6700 xt")
        || name_lower.contains("6700xt")
        || name_lower.contains("6750 xt")
        || name_lower.contains("6750xt")
    {
        return Some("gfx1031");
    }
    if name_lower.contains("6600 xt")
        || name_lower.contains("6600xt")
        || name_lower.contains("6600")
        || name_lower.contains("6650")
    {
        return Some("gfx1032");
    }
    if name_lower.contains("6500 xt") || name_lower.contains("6500xt") {
        return Some("gfx1034");
    }

    // Unknown name → no override
    None
}

/// Parsed lspci line: (pci_slot, name, pci_id)
#[derive(Debug, Clone)]
struct LspciDevice {
    pci_slot: String,
    name: String,
    pci_id: String,
}

/// Parse `lspci -nn` for AMD VGA/3D controllers.
///
/// Returns vec of (pci_slot, marketing_name, pci_id). lspci output format:
/// `BB:DD.F VGA compatible controller [0300]: Vendor Name [Device Name] [vvvv:dddd]`
fn parse_lspci_amd() -> Vec<LspciDevice> {
    let output = match std::process::Command::new("lspci").arg("-nn").output() {
        Ok(o) if o.status.success() => o.stdout,
        _ => return Vec::new(),
    };

    let mut devices = Vec::new();
    let text = String::from_utf8_lossy(&output);

    for line in text.lines() {
        let line = line.trim();
        // Must contain AMD and VGA [0300] or 3D [0302]
        if !line.contains("AMD") && !line.contains("Advanced Micro Devices") {
            continue;
        }
        if !line.contains("[0300]") && !line.contains("[0302]") {
            continue;
        }

        // Extract PCI slot (BB:DD.F format, strip domain prefix if present)
        // lspci format: "0000:BB:DD.F ..." or "BB:DD.F ..."
        let pci_slot = line
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_start_matches("0000:")
            .to_string();
        if pci_slot.is_empty() {
            continue;
        }

        // Find PCI ID bracket: [vvvv:dddd]
        let pci_id = if let Some(bracket_start) = line.rfind('[') {
            let bracket_content = &line[bracket_start + 1..];
            let bracket_end = bracket_content.find(']').unwrap_or(bracket_content.len());
            let id_str = &bracket_content[..bracket_end];
            id_str
                .split(':')
                .nth(1)
                .map(|id| format!("0x{}", id.to_lowercase()))
                .unwrap_or_else(|| "unknown".to_string())
        } else {
            "unknown".to_string()
        };

        // Extract device name: between [0300]: and [vvvv:dddd]
        // Format: "... [0300]: Vendor Name [Device Name] [vvvv:dddd] ..."
        let name = if let Some(class_end) = line.find("]:") {
            let after_class = &line[class_end + 2..];
            if let Some(dev_bracket) = after_class.rfind('[') {
                after_class[..dev_bracket].trim().to_string()
            } else {
                "AMD GPU".to_string()
            }
        } else {
            "AMD GPU".to_string()
        };

        devices.push(LspciDevice {
            pci_slot,
            name,
            pci_id,
        });
    }

    devices
}

/// Check if rocminfo is available and runnable.
fn rocminfo_available() -> bool {
    std::process::Command::new("rocminfo")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Parse rocminfo agent list for ROCm device ordering (optional enrichment).
///
/// Returns vec of (name, gfx_arch) in ROCm device index order. Cross-verified
/// against lspci/sysfs devices by the caller.
fn parse_rocminfo_agents() -> Vec<(String, Option<String>)> {
    let output = match std::process::Command::new("rocminfo").output() {
        Ok(o) if o.status.success() => o.stdout,
        _ => return Vec::new(),
    };

    let mut agents = Vec::new();
    let text = String::from_utf8_lossy(&output);

    // rocminfo sections: "Agent 1" ... "Agent 2" ...
    for agent_block in text.split("****") {
        let mut name = None;
        let mut gfx_arch = None;

        for line in agent_block.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("Name:") {
                name = Some(rest.trim().to_string());
            }
            if let Some(rest) = line.strip_prefix("gfx version:") {
                gfx_arch = Some(rest.trim().to_string());
            }
        }

        if let Some(n) = name {
            agents.push((n, gfx_arch));
        }
    }

    agents
}

/// Detect ALL AMD GPUs via lspci + sysfs (includes iGPU; use
/// [`detect_discrete_amd_gpus`] to filter).
///
/// **Primary source:** `lspci -nn` (universally available, no rocminfo needed).
/// Parses AMD VGA `[0300]` / 3D `[0302]` lines to extract PCI slot, device NAME,
/// and PCI ID `[1002:xxxx]`. This provides reliable device names without rocminfo.
///
/// **Per-device enrichment:**
/// - PCI slot: from lspci
/// - Marketing name: from lspci (e.g. "Navi 31 [Radeon RX 7900 XT/...]")
/// - PCI ID: from lspci `[1002:xxxx]` format
/// - VRAM: from sysfs `mem_info_vram_total` via [`read_vram_bytes`]
/// - gfx arch: from PCI ID table ([`DISCRETE_PCI_ID_TO_GFX`]), or marketing
///   name fallback ([`gfx_from_marketing_name`]), or None for unknown
///
/// **iGPU classification:** Uses name+ID tandem via [`device_is_integrated`]
/// with BOTH marketing_name (from lspci) AND pci_id. A device is integrated only
/// if all signals agree (PCI ID, name, gfx arch, VRAM). Verified dGPU iff
/// `!is_integrated`.
///
/// **rocminfo enrichment (optional):** If `rocminfo` is on PATH and runs,
/// parses its agent list for authoritative ROCm device names + gfx archs and
/// cross-verifies against lspci/sysfs devices (match by name/PCI). Uses rocminfo's
/// device ordering to inform `HIP_VISIBLE_DEVICES` when available; else falls
/// back to verified lspci/sysfs dGPU list in pci-slot order. rocminfo NEVER
/// required — detection works fully without it.
///
/// Returns all AMD GPUs (iGPU + dGPU), each with verified name+PCI_ID+gfx+VRAM.
pub fn detect_amd_gpus() -> Vec<AmdGpu> {
    use std::collections::{HashMap, HashSet};

    let mut gpus = Vec::new();
    let mut seen_slots = HashSet::new();

    // Step 1: Parse lspci for AMD GPU names + PCI IDs (PRIMARY source)
    let lspci_devices = parse_lspci_amd();
    let mut lspci_map: HashMap<String, LspciDevice> = HashMap::new();
    for dev in lspci_devices {
        lspci_map.insert(dev.pci_slot.clone(), dev);
    }

    // Step 2: Walk sysfs /sys/class/drm/card*/device for VRAM + cross-verification
    let entries = match std::fs::read_dir("/sys/class/drm") {
        Ok(e) => e,
        Err(_) => return gpus,
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }

        let device_path = entry.path().join("device");
        let canonical = match device_path.canonicalize() {
            Ok(p) => p,
            Err(_) => continue,
        };

        if !seen_slots.insert(canonical.clone()) {
            continue;
        }

        // Read vendor (must be AMD 0x1002)
        let vendor_path = canonical.join("vendor");
        let vendor = match std::fs::read_to_string(&vendor_path) {
            Ok(v) => v.trim().to_lowercase(),
            Err(_) => continue,
        };
        if !vendor.starts_with("0x1002") {
            continue;
        }

        // Read class — VGA (0x030000) or 3D (0x030200) ONLY
        let class_path = canonical.join("class");
        let class = match std::fs::read_to_string(&class_path) {
            Ok(c) => c.trim().to_lowercase(),
            Err(_) => continue,
        };
        let class_val = match class.strip_prefix("0x") {
            Some(hex) => match u32::from_str_radix(hex, 16) {
                Ok(v) => v,
                Err(_) => continue,
            },
            None => continue,
        };
        const PCI_CLASS_VGA: u32 = 0x030000;
        const PCI_CLASS_3D: u32 = 0x030200;
        const CLASS_MASK: u32 = 0xFFFF00;
        let masked_class = class_val & CLASS_MASK;
        if masked_class != PCI_CLASS_VGA && masked_class != PCI_CLASS_3D {
            continue;
        }

        // Read PCI device ID from sysfs
        let device_path_id = canonical.join("device");
        let sysfs_pci_id = match std::fs::read_to_string(&device_path_id) {
            Ok(d) => d.trim().to_string(),
            Err(_) => continue,
        };

        // Read uevent for PCI_SLOT_NAME
        let uevent_path = canonical.join("uevent");
        let pci_slot = match std::fs::read_to_string(&uevent_path) {
            Ok(content) => content
                .lines()
                .find(|line| line.starts_with("PCI_SLOT_NAME="))
                .and_then(|line| line.split('=').nth(1))
                .map(|s| s.trim_start_matches("0000:").to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            Err(_) => "unknown".to_string(),
        };

        // Read VRAM total from sysfs
        let vram_bytes = read_vram_bytes(&pci_slot);

        // Get marketing name + PCI ID from lspci (primary source)
        let lspci_dev = lspci_map.get(&pci_slot);
        let marketing_name = lspci_dev
            .as_ref()
            .map(|d| d.name.clone())
            .filter(|n| !n.is_empty());
        let lspci_pci_id = lspci_dev
            .as_ref()
            .map(|d| d.pci_id.clone())
            .filter(|i| i != "unknown");

        // Prefer lspci PCI ID (parsed from [1002:xxxx]), fall back to sysfs
        let pci_device_id = lspci_pci_id.unwrap_or_else(|| sysfs_pci_id.clone());

        // Resolve gfx arch: PCI table first, marketing name fallback
        let mut gfx_arch = pci_id_to_gfx(&pci_device_id).map(|s| s.to_string());
        if gfx_arch.is_none() {
            if let Some(ref name) = marketing_name {
                gfx_arch = gfx_from_marketing_name(name).map(|s| s.to_string());
            }
        }

        // Classify integrated/discrete using NAME + PCI_ID tandem (never ID alone)
        let is_integrated = device_is_integrated(
            marketing_name.as_deref(),
            Some(&pci_device_id),
            gfx_arch.as_deref(),
            vram_bytes,
        );

        gpus.push(AmdGpu {
            pci_slot,
            pci_device_id,
            marketing_name,
            gfx_arch,
            vram_bytes,
            is_integrated,
        });
    }

    // Step 3 (optional): rocminfo enrichment for gfx arch + ROCm device ordering
    if rocminfo_available() {
        let rocminfo_agents = parse_rocminfo_agents();
        if !rocminfo_agents.is_empty() {
            // Cross-verify rocminfo agents against detected devices by name/PCI
            // Enrich gfx arch from rocminfo when available
            for (rocm_name, rocm_gfx) in rocminfo_agents {
                for gpu in &mut gpus {
                    if let Some(ref name) = gpu.marketing_name {
                        // Match by name substring (rocminfo names are shorter)
                        if name.contains(&rocm_name) || rocm_name.contains(name) {
                            if let Some(ref rg) = rocm_gfx {
                                gpu.gfx_arch = Some(rg.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    gpus
}

/// Detect ONLY discrete AMD GPUs (iGPU filtered out).
///
/// Filters [`detect_amd_gpus`] to `!is_integrated`, sorted by `pci_slot` for
/// stable indexing (e.g. `HIP_VISIBLE_DEVICES` order).
pub fn detect_discrete_amd_gpus() -> Vec<AmdGpu> {
    let mut dgpus = detect_amd_gpus()
        .into_iter()
        .filter(|gpu| !gpu.is_integrated)
        .collect::<Vec<_>>();
    dgpus.sort_by(|a, b| a.pci_slot.cmp(&b.pci_slot));
    dgpus
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

    // ===========================================================================
    // Detector tests (PCI ID → gfx, name → gfx, class filter)
    // ===========================================================================

    #[test]
    fn pci_id_to_gfx_maps_all_table_entries() {
        // RDNA4
        assert_eq!(pci_id_to_gfx("0x7550"), Some("gfx1201"));
        assert_eq!(pci_id_to_gfx("0x7551"), Some("gfx1201"));
        assert_eq!(pci_id_to_gfx("0x7590"), Some("gfx1200"));
        // RDNA3 Navi31
        assert_eq!(pci_id_to_gfx("0x744c"), Some("gfx1100"));
        assert_eq!(pci_id_to_gfx("0x7448"), Some("gfx1100"));
        assert_eq!(pci_id_to_gfx("0x745e"), Some("gfx1100"));
        // RDNA3 Navi32
        assert_eq!(pci_id_to_gfx("0x747e"), Some("gfx1101"));
        assert_eq!(pci_id_to_gfx("0x7460"), Some("gfx1101"));
        // RDNA3 Navi33
        assert_eq!(pci_id_to_gfx("0x7480"), Some("gfx1102"));
        assert_eq!(pci_id_to_gfx("0x73f0"), Some("gfx1102"));
        // RDNA2 Navi21
        assert_eq!(pci_id_to_gfx("0x73bf"), Some("gfx1030"));
        assert_eq!(pci_id_to_gfx("0x73a1"), Some("gfx1030"));
        // RDNA2 Navi22
        assert_eq!(pci_id_to_gfx("0x73df"), Some("gfx1031"));
        assert_eq!(pci_id_to_gfx("0x73c3"), Some("gfx1031"));
        // RDNA2 Navi23
        assert_eq!(pci_id_to_gfx("0x73ff"), Some("gfx1032"));
        assert_eq!(pci_id_to_gfx("0x73e3"), Some("gfx1032"));
        // RDNA2 Navi24
        assert_eq!(pci_id_to_gfx("0x743f"), Some("gfx1034"));
        assert_eq!(pci_id_to_gfx("0x7421"), Some("gfx1034"));
        // Unknown ID → None
        assert_eq!(pci_id_to_gfx("0x9999"), None);
        assert_eq!(pci_id_to_gfx("0x164e"), None); // iGPU ID not in dGPU table
    }

    #[test]
    fn pci_id_to_gfx_normalizes_case_and_prefix() {
        assert_eq!(pci_id_to_gfx("0x744c"), Some("gfx1100"));
        assert_eq!(pci_id_to_gfx("744C"), Some("gfx1100"));
        assert_eq!(pci_id_to_gfx("744c"), Some("gfx1100"));
        assert_eq!(pci_id_to_gfx("0X744C"), Some("gfx1100"));
    }

    #[test]
    fn gfx_from_marketing_name_handles_navi_patterns() {
        // Generic "Navi NN"
        assert_eq!(gfx_from_marketing_name("Navi 48"), Some("gfx1201"));
        assert_eq!(gfx_from_marketing_name("navi44"), Some("gfx1200"));
        assert_eq!(gfx_from_marketing_name("Navi 31"), Some("gfx1100"));
        assert_eq!(gfx_from_marketing_name("Navi 32"), Some("gfx1101"));
        assert_eq!(gfx_from_marketing_name("Navi 33"), Some("gfx1102"));
        assert_eq!(gfx_from_marketing_name("Navi 21"), Some("gfx1030"));
        assert_eq!(gfx_from_marketing_name("Navi 22"), Some("gfx1031"));
        assert_eq!(gfx_from_marketing_name("Navi 23"), Some("gfx1032"));
        assert_eq!(gfx_from_marketing_name("Navi 24"), Some("gfx1034"));
        // Unknown name → None (no fallback)
        assert_eq!(gfx_from_marketing_name("Generic GPU"), None);
    }

    #[test]
    fn gfx_from_marketing_name_handles_rdnax_names() {
        // RDNA3
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 7900 XTX"),
            Some("gfx1100")
        );
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 7900 GRE"),
            Some("gfx1100")
        );
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 7800 XT"),
            Some("gfx1101")
        );
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 7700 XT"),
            Some("gfx1101")
        );
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 7600 XT"),
            Some("gfx1102")
        );
        // RDNA4
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 9070 XT"),
            Some("gfx1201")
        );
        assert_eq!(gfx_from_marketing_name("Radeon RX 9060"), Some("gfx1200"));
        // RDNA2
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 6900 XT"),
            Some("gfx1030")
        );
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 6800 XT"),
            Some("gfx1030")
        );
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 6700 XT"),
            Some("gfx1031")
        );
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 6600 XT"),
            Some("gfx1032")
        );
        assert_eq!(
            gfx_from_marketing_name("Radeon RX 6500 XT"),
            Some("gfx1034")
        );
    }

    #[test]
    fn class_filter_excludes_non_gpu_classes() {
        // Mock class strings → should be filtered
        // Bridge (0x0604) → NOT a GPU
        let class_bridge = "0x060400";
        assert_eq!(
            u32::from_str_radix(class_bridge.strip_prefix("0x").unwrap(), 16).unwrap() & 0xFFFF00,
            0x060400
        );
        // Audio (0x0403) → NOT a GPU
        let class_audio = "0x040300";
        assert_eq!(
            u32::from_str_radix(class_audio.strip_prefix("0x").unwrap(), 16).unwrap() & 0xFFFF00,
            0x040300
        );
        // VGA (0x030000) → IS a GPU
        let class_vga = "0x030000";
        assert_eq!(
            u32::from_str_radix(class_vga.strip_prefix("0x").unwrap(), 16).unwrap() & 0xFFFF00,
            0x030000
        );
        // 3D (0x030200) → IS a GPU
        let class_3d = "0x030200";
        assert_eq!(
            u32::from_str_radix(class_3d.strip_prefix("0x").unwrap(), 16).unwrap() & 0xFFFF00,
            0x030200
        );
        // Detectors ONLY accept VGA/3D — bridge/audio are excluded
    }
}
