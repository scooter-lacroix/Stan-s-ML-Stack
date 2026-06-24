//! NVIDIA/CUDA exclusion — the SINGLE source of truth for the hard prime
//! requirement that NO NVIDIA/CUDA dependency is ever installed.
//!
//! # Tenet (v0.3.0 remediation)
//!
//! "Nvidia CUDA deps are NEVER installed by ANY component — ESPECIALLY Rusty
//! Llama. THIS IS A HARD PRIME REQUIREMENT. NOT A SINGLE NVIDIA DEP MAY EVER BE
//! INSTALLED. Rusty is for AMD GPUs."
//!
//! # Why this module exists
//!
//! The crate previously had THREE divergent CUDA/nvidia blocklists:
//! - `installer.rs::filter_cuda_requirements` (requirements-file filter, narrow)
//! - `megatron.rs::is_safe_package` (single-package check, broadest)
//! - `textgen.rs::EXCLUDED_PATTERNS` (regex form)
//!
//! They disagreed (e.g. the first lacks `cudnn`/`cublas`/`nvjitlink`/`nccl`
//! that the second catches), and `comfyui` used NONE of them. Everything now
//! routes through here so the rule is defined once and applied everywhere.
//!
//! # Enforcement layers
//!
//! This module is the *detection + filtering* layer. It feeds two enforcement
//! mechanisms used together:
//! 1. **Requirements filtering** — [`filter_requirements`] strips banned lines
//!    from `requirements.txt` files before `pip install -r`.
//! 2. **`--no-deps` + explicit deps** (Stage 3) — for torch-adjacent packages
//!    whose transitive deps could pull `nvidia-*` wheels; installed with
//!    `--no-deps` and a curated dependency list. pip *constraints* cannot
//!    cleanly "ban" a package, so `--no-deps` is the transitive guarantee.
//! 3. **Post-install contamination check** — [`contaminated_packages`] scans
//!    `pip list` output so verification (Stage 4) can raise
//!    `InstallerError::NvidiaContamination`.

/// Package-name prefixes that mark an NVIDIA/CUDA dependency.
///
/// Matched case-insensitively against the normalized package name (prefix match).
/// `triton` is a prefix so it also catches `triton-windows`, `triton[...]`, etc.
pub const BLOCKED_PREFIXES: &[&str] = &[
    "nvidia",
    "cuda",
    "cudnn",
    "cublas",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "cusparselt",
    "nccl",
    "nvtx",
    "nvjitlink",
    "nv-pic",
    "tensorrt",
    "triton",
    "pytorch-cuda",
    "torch-cuda",
];

/// Exact package names (after normalization) that are NVIDIA/CUDA/ROCm-conflicting.
///
/// These must come from the ROCm build managed by Rusty, never from a CUDA
/// default wheel.
pub const BLOCKED_EXACT: &[&str] = &[
    "torch",
    "torchvision",
    "torchaudio",
    "xformers",
    "flash-attn",
    "flash_attn",
];

/// CUDA wheel URL markers (in URLs like `...+cu124...whl`).
pub const CUDA_URL_MARKERS: &[&str] = &[
    "+cu1", "+cu2", "+cu3", "+cu4", "cu118", "cu121", "cu124", "cu126", "cu128", "cu129",
];

/// Normalize a requirement spec to its bare package name.
///
/// Strips environment markers (`;`), extras (`[...]`), version specifiers
/// (`==`, `>=`, etc.), and trailing whitespace. `"nvidia-cudnn-cu12==8.9..."`
/// → `"nvidia-cudnn-cu12"`; `"triton-windows [test]>=3.5 ; sys_platform==..."`
/// → `"triton-windows"`.
pub fn extract_pkg_name(requirement: &str) -> &str {
    let req = requirement.trim();
    let after_marker = req.split(';').next().unwrap_or(req);
    let after_extras = after_marker.split('[').next().unwrap_or(after_marker);
    after_extras
        .split(&['=', '<', '>', '~', '!', ' ', '\t'][..])
        .next()
        .unwrap_or(after_extras)
        .trim()
}

/// Is this package name an NVIDIA/CUDA/ROCm-conflicting package?
///
/// Use on a normalized name (see [`extract_pkg_name`]) or a raw requirement —
/// this normalizes first. Case-insensitive.
pub fn is_cuda_nvidia_package(requirement: &str) -> bool {
    let pkg = extract_pkg_name(requirement).to_lowercase();
    if pkg.is_empty() {
        return false;
    }
    if BLOCKED_EXACT
        .iter()
        .any(|exact| exact.eq_ignore_ascii_case(&pkg))
    {
        return true;
    }
    if BLOCKED_PREFIXES
        .iter()
        .any(|prefix| pkg.starts_with(prefix))
    {
        return true;
    }
    // cupy-cuda*, any name containing "cuda" (e.g. cupy-cuda12, torch-cuda).
    if pkg.contains("cuda") {
        return true;
    }
    false
}

/// Does this wheel URL carry a CUDA build marker? (e.g. `...+cu124...whl`)
pub fn is_cuda_wheel_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    CUDA_URL_MARKERS.iter().any(|marker| lower.contains(marker))
        || (lower.contains("+cu") && lower.contains(".whl"))
}

/// NVIDIA/CUDA *runtime-library* prefixes (the `nvidia-*`/`cuda*` wheels), for
/// the **execution chokepoint**. Deliberately EXCLUDES the torch/triton family
/// — those are legitimately installed from the ROCm index by the core
/// pytorch/triton installers, and are instead kept out of third-party installs
/// by [`is_cuda_nvidia_package`] (requirements filtering) + `--no-deps`.
pub const NVIDIA_RUNTIME_PREFIXES: &[&str] = &[
    "nvidia",
    "cuda",
    "cudnn",
    "cublas",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "cusparselt",
    "nccl",
    "nvtx",
    "nvjitlink",
    "nv-pic",
    "tensorrt",
    "pytorch-cuda",
    "torch-cuda",
];

/// Is this an NVIDIA/CUDA *runtime* package (a `nvidia-*`/`cuda*` lib)?
///
/// Narrower than [`is_cuda_nvidia_package`]: returns `false` for
/// torch/torchvision/torchaudio/triton/xformers/flash-attn (core, ROCm-managed).
/// Used by the pip execution chokepoint to block any nvidia/cuda runtime wheel
/// from *any* installer, while still permitting the core pytorch/triton install.
pub fn is_nvidia_cuda_runtime_package(requirement: &str) -> bool {
    let pkg = extract_pkg_name(requirement).to_lowercase();
    if pkg.is_empty() {
        return false;
    }
    if NVIDIA_RUNTIME_PREFIXES
        .iter()
        .any(|prefix| pkg.starts_with(prefix))
    {
        return true;
    }
    // cupy-cuda*, any name containing "cuda".
    if pkg.contains("cuda") {
        return true;
    }
    false
}

/// Is this a CUDA-only source line (e.g. `ik_llama` builds)?
fn is_cuda_only_source(line_lower: &str) -> bool {
    // ik_llama ships CUDA-only wheels.
    if line_lower.contains("ik_llama") {
        return true;
    }
    // exllamav3 / flash_attn wheels with CUDA markers.
    if (line_lower.contains("exllamav3") || line_lower.contains("flash_attn"))
        && (line_lower.contains("+cu") || line_lower.contains("cu1"))
    {
        return true;
    }
    false
}

/// Filter a requirements FILE, returning the content with all
/// NVIDIA/CUDA/ROCm-conflicting lines removed.
///
/// Read-safe: returns an empty string if the file is unreadable. Keeps
/// ordering; drops comments/blank lines. This is the unified replacement for
/// the former `installer.rs::filter_cuda_requirements`.
pub fn filter_requirements_file(path: &str) -> String {
    match std::fs::read_to_string(path) {
        Ok(c) => filter_requirements(&c),
        Err(_) => String::new(),
    }
}

/// Filter requirements TEXT, returning lines that are safe to install on a
/// ROCm/AMD stack (all NVIDIA/CUDA/ROCm-conflicting lines removed).
pub fn filter_requirements(content: &str) -> String {
    let mut kept = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if is_cuda_nvidia_package(trimmed) {
            continue;
        }
        let line_lower = trimmed.to_lowercase();
        if line_lower.contains("http://") || line_lower.contains("https://") {
            if is_cuda_wheel_url(trimmed) || is_cuda_only_source(&line_lower) {
                continue;
            }
        } else if is_cuda_only_source(&line_lower) {
            continue;
        }
        kept.push(line.to_string());
    }
    kept.join("\n")
}

/// Scan `pip list` output for any installed NVIDIA/CUDA packages.
///
/// Returns the offending package display lines (for the
/// `InstallerError::NvidiaContamination` payload and verification reports).
pub fn contaminated_packages(pip_list_output: &str) -> Vec<String> {
    let mut found = Vec::new();
    for line in pip_list_output.lines() {
        let name = line.split_whitespace().next().unwrap_or("");
        if is_cuda_nvidia_package(name) {
            found.push(line.trim().to_string());
        }
    }
    found
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_wheels_are_blocked() {
        assert!(is_cuda_nvidia_package("nvidia-cudnn-cu12"));
        assert!(is_cuda_nvidia_package("nvidia-cublas-cu12==12.1.3.1"));
        assert!(is_cuda_nvidia_package("nccl"));
        assert!(is_cuda_nvidia_package("nvjitlink"));
        assert!(is_cuda_nvidia_package("cuda-python"));
        assert!(is_cuda_nvidia_package("cupy-cuda12x"));
        assert!(is_cuda_nvidia_package("tensorrt"));
    }

    #[test]
    fn torch_family_is_blocked() {
        assert!(is_cuda_nvidia_package("torch"));
        assert!(is_cuda_nvidia_package("torchvision>=0.17"));
        assert!(is_cuda_nvidia_package("torchaudio"));
        assert!(is_cuda_nvidia_package("triton"));
        assert!(is_cuda_nvidia_package("triton-windows==3.5.1"));
        assert!(is_cuda_nvidia_package("xformers"));
        assert!(is_cuda_nvidia_package("flash-attn"));
    }

    #[test]
    fn safe_amd_packages_pass() {
        assert!(!is_cuda_nvidia_package("deepspeed"));
        assert!(!is_cuda_nvidia_package("transformers"));
        assert!(!is_cuda_nvidia_package("accelerate"));
        assert!(!is_cuda_nvidia_package("einops"));
        assert!(!is_cuda_nvidia_package("numpy"));
        assert!(!is_cuda_nvidia_package("bitsandbytes"));
        assert!(!is_cuda_nvidia_package("vllm"));
        assert!(!is_cuda_nvidia_package("wandb"));
    }

    #[test]
    fn cuda_wheel_urls_detected() {
        assert!(is_cuda_wheel_url(
            "https://x/llama_cpp_binaries-0.124.0+cu124-py3.whl"
        ));
        assert!(is_cuda_wheel_url("https://x/torch-2.4.0+cu118.whl"));
        assert!(is_cuda_wheel_url("https://x/flash_attn-2.6+cu128.whl"));
        assert!(!is_cuda_wheel_url("https://x/torch-2.4.0+rocm624.whl"));
        assert!(!is_cuda_wheel_url("https://x/numpy-1.26.whl"));
    }

    #[test]
    fn filter_requirements_strips_banned_lines() {
        let reqs = "\
# comment
torch==2.4.0
nvidia-cudnn-cu12
numpy>=1.26
transformers
https://download.pytorch.org/whl/cu121/xformers-0.0.27%2Bcu121.whl
ik_llama_cpp_cuda-0.1.whl
deepspeed
";
        let filtered = filter_requirements(reqs);
        assert!(!filtered.contains("torch"));
        assert!(!filtered.contains("nvidia-cudnn"));
        assert!(!filtered.contains("xformers"));
        assert!(!filtered.contains("ik_llama"));
        assert!(filtered.contains("numpy>=1.26"));
        assert!(filtered.contains("transformers"));
        assert!(filtered.contains("deepspeed"));
        assert!(!filtered.contains("# comment"));
    }

    #[test]
    fn extract_pkg_name_handles_specs() {
        assert_eq!(
            extract_pkg_name("nvidia-cudnn-cu12==8.9.2.26"),
            "nvidia-cudnn-cu12"
        );
        assert_eq!(
            extract_pkg_name("triton-windows [test]>=3.5.1 ; sys_platform == 'win32'"),
            "triton-windows"
        );
        assert_eq!(extract_pkg_name("  torch  "), "torch");
    }

    #[test]
    fn contamination_scan_finds_nvidia() {
        let pip_list = "\
Package        Version
-------------  -------
deepspeed      0.14.5
nvidia-cublas-cu12  12.1.3.1
torch          2.4.0
numpy          1.26.4
";
        let found = contaminated_packages(pip_list);
        assert_eq!(found.len(), 2);
        assert!(found.iter().any(|l| l.starts_with("nvidia-cublas")));
        assert!(found.iter().any(|l| l.starts_with("torch")));
        assert!(!found.iter().any(|l| l.contains("deepspeed")));
    }
}
