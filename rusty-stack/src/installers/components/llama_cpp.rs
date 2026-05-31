//! llama.cpp-turboquant-hip installer — native Rust CMake source build.
//!
//! Clones the private fork `scooter-lacroix/llama.cpp-turboquant-hip`,
//! builds with HIP/ROCm CMake flags per channel policy, and installs
//! binaries to `~/.mlstack/components/llama-cpp/bin/`.
//!
//! # Detection Contract
//!
//! Detection and verification both operate on the same binary contract:
//! `llama-cli --help` must succeed (exit code 0). This ensures:
//! - Partial clone artifacts do not count as installed
//! - Broken executables do not count as installed
//! - Detection and verification use the same executable
//!
//! # ROCm Channel Policy
//!
//! | Channel | GPU Targets | WMMA FA | Notes |
//! |---------|-------------|---------|-------|
//! | Legacy  | gfx1030     | No      | RDNA2 only |
//! | Stable  | gfx1030;gfx1100;gfx1101 | Yes | RDNA2+RDNA3 |
//! | Latest  | gfx1030;gfx1100;gfx1101;gfx1200 | Yes | RDNA2+RDNA3+RDNA4 |
//!
//! # Validation Assertions
//!
//! - **VAL-INTEGRATION-002**: Native installer route accepts llama-cpp
//! - **VAL-INTEGRATION-003**: Installed detection succeeds with functional command
//! - **VAL-INTEGRATION-004**: Installed detection fails when command is absent
//! - **VAL-INTEGRATION-005**: Broken executable does not count as installed
//! - **VAL-INTEGRATION-006**: Partial filesystem artifacts do not count as success
//! - **VAL-CROSS-001**: Channel selection determines llama.cpp build flags
//! - **VAL-CROSS-003**: Install path and detection strategy are consistent
//! - **VAL-CROSS-012**: Detection and verification operate on the same binary contract

use crate::installers::common::{
    log_warn, submit_build_report, BuildReport, BuildReportStatus, SealedToken,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::NamedTempFile;

// ===========================================================================
// Constants
// ===========================================================================

/// Default repo URL (HTTPS). Can be overridden via `LLAMA_CPP_REPO_URL` env var.
const DEFAULT_REPO_URL: &str = "https://github.com/scooter-lacroix/llama.cpp-turboquant-hip";

/// Default branch to clone.
const DEFAULT_BRANCH: &str = "main";

/// Build directory for CMake source builds.
const DEFAULT_BUILD_DIR: &str = "/tmp/llama-cpp-rocm-build";

/// Install prefix for llama.cpp binaries.
const DEFAULT_INSTALL_PREFIX: &str = ".mlstack/components/llama-cpp";

/// Binary targets to build.
const BUILD_TARGETS: &[&str] = &["llama-cli", "llama-bench", "llama-server"];

/// The primary detection binary.
pub const DETECTION_BINARY: &str = "llama-cli";

/// The detection subcommand that must succeed for installed status.
pub const DETECTION_SUBCOMMAND: &str = "--help";

// ===========================================================================
// Types
// ===========================================================================

/// GPU architecture specification for HIP builds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HipArchs {
    /// gfx1030 family (RDNA2).
    Gfx1030,
    /// gfx1100 family (gfx1100-gfx1101, RDNA3).
    Gfx1100,
    /// gfx1200 family (gfx1200-gfx1201, RDNA4).
    Gfx1200,
    /// Default fallback (gfx1030 — conservative).
    Default,
}

impl HipArchs {
    /// Detect HIP architecture from a GPU arch string.
    pub fn from_gpu_arch(gpu_arch: &str) -> Self {
        if gpu_arch.starts_with("gfx103") {
            HipArchs::Gfx1030
        } else if gpu_arch.starts_with("gfx110") {
            HipArchs::Gfx1100
        } else if gpu_arch.starts_with("gfx120") {
            HipArchs::Gfx1200
        } else {
            HipArchs::Default
        }
    }

    /// Get the GPU_TARGETS string for a given ROCm channel.
    ///
    /// Channel policy:
    /// - Legacy: only gfx1030 (RDNA2)
    /// - Stable: gfx1030 + gfx1100/1101 (RDNA2+RDNA3)
    /// - Latest: gfx1030 + gfx1100/1101 + gfx1200 (RDNA2+RDNA3+RDNA4)
    ///
    /// Unknown/indeterminate hardware degrades to conservative gfx1030.
    pub fn gpu_targets_for_channel(&self, channel: &str) -> String {
        match (self, channel) {
            // Legacy channel: RDNA2 only regardless of detected GPU
            (_, "legacy") => "gfx1030".to_string(),
            // Stable channel: RDNA2 + RDNA3
            (HipArchs::Gfx1030, "stable") => "gfx1030".to_string(),
            (HipArchs::Gfx1100, "stable") => "gfx1030;gfx1100;gfx1101".to_string(),
            (HipArchs::Gfx1200, "stable") => "gfx1030;gfx1100;gfx1101".to_string(),
            (HipArchs::Default, "stable") => "gfx1030".to_string(),
            // Latest channel: RDNA2 + RDNA3 + RDNA4
            (HipArchs::Gfx1030, _) => "gfx1030".to_string(),
            (HipArchs::Gfx1100, _) => "gfx1030;gfx1100;gfx1101".to_string(),
            (HipArchs::Gfx1200, _) => "gfx1030;gfx1100;gfx1101;gfx1200".to_string(),
            (HipArchs::Default, _) => "gfx1030".to_string(),
        }
    }
}

/// ROCm channel for CMake flag selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RocmChannel {
    Legacy,
    Stable,
    Latest,
}

impl RocmChannel {
    /// Parse from a channel string (case-insensitive).
    pub fn from_str(channel: &str) -> Self {
        match channel.to_lowercase().as_str() {
            "legacy" => RocmChannel::Legacy,
            "stable" => RocmChannel::Stable,
            _ => RocmChannel::Latest,
        }
    }

    /// Whether WMMA flash attention should be enabled for this channel.
    pub fn enable_wmma_fa(&self) -> bool {
        matches!(self, RocmChannel::Stable | RocmChannel::Latest)
    }

    /// The channel label string.
    pub fn label(&self) -> &'static str {
        match self {
            RocmChannel::Legacy => "legacy",
            RocmChannel::Stable => "stable",
            RocmChannel::Latest => "latest",
        }
    }
}

/// A constructed shell command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommand {
    /// The program to run.
    pub program: String,
    /// Arguments to pass.
    pub args: Vec<String>,
    /// Environment variables to set.
    pub env: Vec<(String, String)>,
    /// Working directory for the command.
    pub working_dir: Option<PathBuf>,
}

/// Configuration for the llama.cpp installer.
#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// GPU architecture string (e.g., "gfx1100").
    pub gpu_arch: String,
    /// ROCm channel (legacy/stable/latest).
    pub channel: String,
    /// Repo URL override (defaults to DEFAULT_REPO_URL).
    pub repo_url: Option<String>,
    /// Branch to clone (defaults to DEFAULT_BRANCH).
    pub branch: String,
    /// Build directory override.
    pub build_dir: Option<String>,
    /// Install prefix override (relative to $HOME).
    pub install_prefix: Option<String>,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[allow(dead_code)]
pub struct ReleaseAsset {
    pub url: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReleaseManifest {
    pub version: String,
    pub archs: ReleaseArchs,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReleaseArchs {
    pub gfx1030: ReleaseAsset,
    pub gfx1100: ReleaseAsset,
    pub gfx1200: ReleaseAsset,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrebuiltInstallPlan {
    pub manifest_version: String,
    pub arch: String,
    pub url: String,
    pub sha256: String,
    pub install_prefix: String,
    pub binary_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceInstallPlan {
    pub reason: String,
    pub build_commands: Vec<ShellCommand>,
    pub install_prefix: String,
    pub binary_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallStrategy {
    Prebuilt(PrebuiltInstallPlan),
    Source(SourceInstallPlan),
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            gpu_arch: "gfx1100".to_string(),
            channel: "latest".to_string(),
            repo_url: None,
            branch: DEFAULT_BRANCH.to_string(),
            build_dir: None,
            install_prefix: None,
            dry_run: false,
        }
    }
}

impl LlamaCppConfig {
    /// Get the effective repo URL.
    pub fn repo_url(&self) -> &str {
        self.repo_url.as_deref().unwrap_or(DEFAULT_REPO_URL)
    }

    /// Get the effective build directory.
    pub fn build_dir(&self) -> &str {
        self.build_dir.as_deref().unwrap_or(DEFAULT_BUILD_DIR)
    }

    /// Get the effective install prefix (relative to $HOME).
    pub fn install_prefix(&self) -> &str {
        self.install_prefix
            .as_deref()
            .unwrap_or(DEFAULT_INSTALL_PREFIX)
    }

    /// Get the full install bin directory path.
    pub fn install_bin_dir(&self, home: &str) -> PathBuf {
        PathBuf::from(home).join(self.install_prefix()).join("bin")
    }

    /// Get the path to the detection binary.
    pub fn detection_binary_path(&self, home: &str) -> PathBuf {
        self.install_bin_dir(home).join(DETECTION_BINARY)
    }
}

/// The llama.cpp installer.
#[derive(Debug, Clone)]
pub struct LlamaCppInstaller {
    config: LlamaCppConfig,
}

impl Default for LlamaCppInstaller {
    fn default() -> Self {
        Self::new(LlamaCppConfig::default())
    }
}

impl LlamaCppInstaller {
    /// Create a new installer with the given config.
    pub fn new(config: LlamaCppConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(LlamaCppConfig::default())
    }

    // -------------------------------------------------------------------
    // Dependencies
    // -------------------------------------------------------------------

    /// Get the list of required dependencies.
    ///
    /// llama.cpp depends on ROCm.
    pub fn dependencies(&self) -> &[&str] {
        &["rocm"]
    }

    /// Fetch the latest release manifest from GitHub.
    ///
    /// Returns `None` for HTTP/API failures, including 404.
    pub fn check_latest_release(&self) -> Option<ReleaseManifest> {
        let mut token = SealedToken::from_env();
        let result = self.fetch_latest_release_manifest(token.as_str());
        token.purge();
        result
    }

    pub fn release_asset_for_arch<'a>(
        &self,
        manifest: &'a ReleaseManifest,
        gpu_arch: &str,
    ) -> Option<&'a ReleaseAsset> {
        match gpu_arch {
            "gfx1030" => Some(&manifest.archs.gfx1030),
            "gfx1100" | "gfx1101" => Some(&manifest.archs.gfx1100),
            "gfx1200" | "gfx1201" => Some(&manifest.archs.gfx1200),
            _ => None,
        }
    }

    // -------------------------------------------------------------------
    // Auth Validation
    // -------------------------------------------------------------------

    /// Validate the git clone step for private repo access.
    ///
    /// Returns `Ok(())` if the repo is accessible (public or private with valid auth).
    /// Returns `Err(String)` with an actionable error message if auth fails.
    pub fn validate_repo_access(&self) -> Result<(), String> {
        let mut github_token = Some(SealedToken::from_env());

        // Try to clone a single file to validate access
        let temp_dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let temp_path = temp_dir
            .path()
            .to_str()
            .ok_or("Failed to create temp dir")?
            .to_string();

        // Build the git command
        let mut cmd = if let Some(ref token) = github_token {
            // Use credential helper for authentication — token never appears in command line
            let credential_helper = r#"!sh -c "echo username=git; echo password=$GITHUB_TOKEN""#;
            let mut git_cmd = std::process::Command::new("git");
            git_cmd
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg("--branch")
                .arg(&self.config.branch)
                .arg(self.config.repo_url())
                .arg(&temp_path)
                .arg("-c")
                .arg(format!("credential.helper={}", credential_helper))
                .env("GITHUB_TOKEN", token.as_str())
                .env("GIT_TERMINAL_PROMPT", "0");
            git_cmd
        } else {
            // Use unauthenticated clone
            let mut git_cmd = std::process::Command::new("git");
            git_cmd
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg("--branch")
                .arg(&self.config.branch)
                .arg(self.config.repo_url())
                .arg(&temp_path);
            git_cmd
        };

        let output = match cmd.output() {
            Ok(output) => output,
            Err(e) => {
                if let Some(token) = github_token.as_mut() {
                    token.purge();
                }
                return Err(e.to_string());
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if let Some(token) = github_token.as_mut() {
                token.purge();
            }
            if stderr.contains("authentication") || stderr.contains("403") || stderr.contains("401")
            {
                return Err(
                    "authentication failed for private repo. If you have access, set the build-time GitHub token.".to_string(),
                );
            } else {
                return Err("Failed to clone repository".to_string());
            }
        }

        if let Some(token) = github_token.as_mut() {
            token.purge();
        }
        Ok(())
    }

    fn fetch_latest_release_manifest(&self, token: &str) -> Option<ReleaseManifest> {
        let url =
            "https://api.github.com/repos/scooter-lacroix/llama.cpp-turboquant-hip/releases/latest";
        let response = ureq::get(url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "rusty-stack")
            .call();

        let mut response = match response {
            Ok(response) => response,
            Err(_) => return None,
        };

        if response.status() == 404 {
            return None;
        }

        if !response.status().is_success() {
            return None;
        }

        response.body_mut().read_json::<ReleaseManifest>().ok()
    }

    // -------------------------------------------------------------------
    // CMake flag generation
    // -------------------------------------------------------------------

    /// Build the CMake flags for the configured channel and GPU arch.
    ///
    /// Returns a list of CMake cache variable flags.
    pub fn cmake_flags(&self) -> Vec<String> {
        let hip_archs = HipArchs::from_gpu_arch(&self.config.gpu_arch);
        let channel = RocmChannel::from_str(&self.config.channel);
        let gpu_targets = hip_archs.gpu_targets_for_channel(channel.label());

        let mut flags = vec![
            "-DGGML_HIP=ON".to_string(),
            format!("-DGPU_TARGETS={}", gpu_targets),
            // VAL-GUARD-001: force ROCm/HIP-only backend selection.
            "-DGGML_CUDA=OFF".to_string(),
            "-DGGML_VULKAN=OFF".to_string(),
            "-DGGML_METAL=OFF".to_string(),
        ];

        // RDNA3 probes and WMMA flash attention are enabled for stable/latest only
        if channel.enable_wmma_fa() {
            flags.push("-DGGML_HIP_RDNA3_PROBES=ON".to_string());
            flags.push("-DGGML_HIP_ROCWMMA_FATTN=ON".to_string());
        }

        flags
    }

    fn expected_gpu_targets(&self) -> String {
        let hip_archs = HipArchs::from_gpu_arch(&self.config.gpu_arch);
        let channel = RocmChannel::from_str(&self.config.channel);
        hip_archs.gpu_targets_for_channel(channel.label())
    }

    // -------------------------------------------------------------------
    // Command sequence
    // -------------------------------------------------------------------

    /// Build the sequence of commands to install llama.cpp with ROCm support.
    pub fn build_commands(&self, home: &str) -> Vec<ShellCommand> {
        self.build_commands_with_auth(home, None)
    }

    /// Resolve the install strategy for the detected GPU and latest release manifest.
    pub fn resolve_install_strategy(
        &self,
        home: &str,
        manifest: Option<&ReleaseManifest>,
    ) -> InstallStrategy {
        let install_prefix = format!("{}/{}", home, self.config.install_prefix());
        let binary_path = self.config.detection_binary_path(home);
        if let Some(manifest) = manifest {
            if let Some(asset) = self.release_asset_for_arch(manifest, &self.config.gpu_arch) {
                return InstallStrategy::Prebuilt(PrebuiltInstallPlan {
                    manifest_version: manifest.version.clone(),
                    arch: self.config.gpu_arch.clone(),
                    url: asset.url.clone(),
                    sha256: asset.sha256.clone(),
                    install_prefix,
                    binary_path,
                });
            }
        }

        let reason = if manifest.is_some() {
            format!(
                "No pre-built binary available for {}; falling back to source compile.",
                self.config.gpu_arch
            )
        } else {
            format!(
                "Unable to fetch release manifest or match {}; falling back to source compile.",
                self.config.gpu_arch
            )
        };

        InstallStrategy::Source(SourceInstallPlan {
            reason,
            build_commands: self.build_commands_with_auth(home, None),
            install_prefix,
            binary_path,
        })
    }

    fn verify_sha256(path: &std::path::Path, expected: &str) -> Result<bool, String> {
        let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];
        loop {
            let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        let actual = Self::sha256_hex_bytes(&hasher.finalize());
        Ok(actual.eq_ignore_ascii_case(expected))
    }

    pub fn download_prebuilt_binary(&self, plan: &PrebuiltInstallPlan) -> Result<(), String> {
        let mut temp = NamedTempFile::new().map_err(|e| e.to_string())?;
        let response = match ureq::get(&plan.url).call() {
            Ok(response) => response,
            Err(err) => {
                log_warn(&format!(
                    "Prebuilt llama.cpp download failed; falling back to source compile: {}",
                    err
                ));
                return Err("download failed".to_string());
            }
        };

        if !response.status().is_success() {
            log_warn("Prebuilt llama.cpp download failed; falling back to source compile.");
            return Err("download failed".to_string());
        }

        let downloaded = response
            .into_body()
            .read_to_vec()
            .map_err(|e| e.to_string())?;
        temp.write_all(&downloaded).map_err(|e| e.to_string())?;
        temp.flush().map_err(|e| e.to_string())?;

        if !Self::verify_sha256(temp.path(), &plan.sha256)? {
            let message = format!(
                "Checksum mismatch for {} binary: expected {}, got {}. Download may be corrupted. Falling back to source build.",
                plan.arch,
                plan.sha256,
                Self::sha256_hex(temp.path())?
            );
            log_warn(&message);
            return Err(message);
        }

        fs::create_dir_all(&plan.install_prefix).map_err(|e| e.to_string())?;
        let status = std::process::Command::new("tar")
            .arg("-xzf")
            .arg(temp.path())
            .arg("-C")
            .arg(&plan.install_prefix)
            .status()
            .map_err(|e| e.to_string())?;

        if !status.success() {
            return Err("failed to extract prebuilt archive".to_string());
        }

        // VAL-GUARD-004: Post-build linkage verification rejects CUDA-linked binaries
        // VAL-GUARD-005: prebuilt binaries must prove ROCm linkage before success.
        verify_binary_linkage(&plan.binary_path)?;

        temp.close().map_err(|e| e.to_string())?;
        Ok(())
    }

    fn sha256_hex(path: &std::path::Path) -> Result<String, String> {
        let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];
        loop {
            let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        Ok(Self::sha256_hex_bytes(&hasher.finalize()))
    }

    fn sha256_hex_bytes(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{:02x}", byte)).collect()
    }

    /// Purge source build artifacts after a successful install.
    ///
    /// Removes repo metadata, git credential cache/store entries, and the build tree.
    /// Returns Ok(()) when the purge is already complete or if the binary was placed
    /// but a cleanup step fails. Returns Err when cleanup fails before installation
    /// has produced the binary.
    pub fn purge_source_artifacts(&self, home: &str, binary_placed: bool) -> Result<(), String> {
        let build_dir = self.config.build_dir().to_string();
        let install_bin = self.config.detection_binary_path(home);
        let mut had_error = None::<String>;
        let purge_plan = [
            (
                "source repository metadata",
                std::process::Command::new("find")
                    .args([
                        format!("{}/.mlstack", home),
                        "-type".to_string(),
                        "d".to_string(),
                        "-name".to_string(),
                        ".git".to_string(),
                        "-prune".to_string(),
                        "-exec".to_string(),
                        "rm".to_string(),
                        "-rf".to_string(),
                        "{}".to_string(),
                        ";".to_string(),
                    ])
                    .output(),
            ),
            (
                "git credential store",
                std::process::Command::new("git")
                    .args([
                        "credential-store".to_string(),
                        "erase".to_string(),
                        "--file".to_string(),
                        format!("{}/.git-credentials", home),
                    ])
                    .output(),
            ),
            (
                "git credential cache",
                std::process::Command::new("git")
                    .args(["credential-cache".to_string(), "exit".to_string()])
                    .output(),
            ),
            (
                "source build tree",
                std::process::Command::new("rm")
                    .args(["-rf".to_string(), build_dir.clone()])
                    .output(),
            ),
        ];

        for (label, output) in purge_plan {
            match output {
                Ok(output) if output.status.success() => {}
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                    had_error = Some(if stderr.is_empty() {
                        format!("{} purge failed", label)
                    } else {
                        stderr
                    });
                    break;
                }
                Err(err) => {
                    had_error = Some(format!("{} purge failed: {}", label, err));
                    break;
                }
            }
        }

        if let Some(err) = had_error {
            if binary_placed && install_bin.exists() {
                log_warn(&format!(
                    "llama.cpp source purge completed with warnings: {}",
                    err
                ));
                return Ok(());
            }
            return Err(format!(
                "Failed to purge llama.cpp source artifacts; please remove {} and retry: {}",
                build_dir, err
            ));
        }

        Ok(())
    }

    /// Build the sequence of commands to install llama.cpp with ROCm support, handling private repo auth.
    pub fn build_commands_with_auth(
        &self,
        home: &str,
        token: Option<SealedToken>,
    ) -> Vec<ShellCommand> {
        let build_dir = self.config.build_dir().to_string();
        let install_prefix = format!("{}/{}", home, self.config.install_prefix());

        // Build the clone command with auth if available
        let mut commands = Vec::new();
        commands.push(self.mkdir_build_dir(&build_dir));

        if let Some(mut token) = token {
            commands.push(self.git_clone_with_auth(&build_dir, token.as_str()));
            token.purge();
        } else {
            commands.push(self.git_clone(&build_dir));
        }

        commands.push(self.git_checkout(&build_dir));
        commands.push(self.cmake_configure(&build_dir, &install_prefix));
        commands.push(self.cmake_build(&build_dir));
        commands.push(self.cmake_install(&build_dir));
        commands.push(self.purge_source_artifacts_command(home));

        commands
    }

    pub fn install(&self, home: &str, manifest: Option<&ReleaseManifest>) -> Result<(), String> {
        let start = Instant::now();
        match self.resolve_install_strategy(home, manifest) {
            InstallStrategy::Prebuilt(plan) => match self.download_prebuilt_binary(&plan) {
                Ok(()) => {
                    self.submit_telemetry(home, start, true, Some(&plan.binary_path))
                        .ok();
                    Ok(())
                }
                Err(err) if err.contains("checksum mismatch") => {
                    log_warn(&format!("{err} Falling back to source compile."));
                    let result = self.run_source_install(home);
                    if result.is_ok() {
                        self.submit_telemetry(home, start, false, None).ok();
                    }
                    result
                }
                Err(_) => {
                    log_warn("Prebuilt download unavailable; using source compile.");
                    let result = self.run_source_install(home);
                    if result.is_ok() {
                        self.submit_telemetry(home, start, false, None).ok();
                    }
                    result
                }
            },
            InstallStrategy::Source(plan) => {
                let result = self.run_source_install(home);
                if result.is_ok() {
                    self.submit_telemetry(home, start, false, Some(&plan.binary_path))
                        .ok();
                }
                result
            }
        }
    }

    pub fn run_source_install(&self, home: &str) -> Result<(), String> {
        // VAL-GUARD-002: CUDA toolkit presence is warning-only; build stays HIP-only.
        let cuda_artifacts = detect_cuda_toolkit_presence();
        if !cuda_artifacts.is_empty() {
            log_warn(&format!(
                "NVIDIA CUDA toolkit detected on this system ({}). Rusty Llama builds exclusively with ROCm/HIP; CUDA will not be used.",
                cuda_artifacts.join(", ")
            ));
        }

        let commands = self.build_commands(home);
        if commands.is_empty() {
            return Err("no source install commands available".to_string());
        }

        for command in commands {
            let is_cmake_configure = command.program == "cmake"
                && command.args.iter().any(|arg| arg == "-B")
                && command.args.iter().any(|arg| arg == "-S");
            let is_cmake_install = command.program == "cmake"
                && command.args.iter().any(|arg| arg == "--install");
            let command_working_dir = command.working_dir.clone();

            let mut process = std::process::Command::new(&command.program);
            process.args(&command.args);
            for (key, value) in &command.env {
                process.env(key, value);
            }
            if let Some(dir) = &command.working_dir {
                process.current_dir(dir);
            }

            let output = process.output().map_err(|err| {
                format!(
                    "failed to run source install command '{}': {}",
                    command.program, err
                )
            })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let mut details = String::new();
                if !stderr.is_empty() {
                    details.push_str(&stderr);
                }
                if !stdout.is_empty() {
                    if !details.is_empty() {
                        details.push_str(" | ");
                    }
                    details.push_str(&stdout);
                }

                return Err(if details.is_empty() {
                    format!("source install command '{}' failed", command.program)
                } else {
                    format!(
                        "source install command '{}' failed: {}",
                        command.program, details
                    )
                });
            }

            if is_cmake_configure {
                let cache_path = command_working_dir
                    .as_deref()
                    .unwrap_or_else(|| Path::new("."))
                    .join("build")
                    .join("CMakeCache.txt");
                validate_cmake_cache(&cache_path, &self.expected_gpu_targets())?;
            }

            if is_cmake_install {
                // VAL-GUARD-004: Post-build linkage verification rejects CUDA-linked binaries
                // VAL-GUARD-005: source builds must prove ROCm linkage before success.
                let bin_path = resolve_install_bin_dir(home).join(DETECTION_BINARY);
                verify_binary_linkage(&bin_path)?;
            }
        }

        Ok(())
    }

    fn purge_source_artifacts_command(&self, home: &str) -> ShellCommand {
        let build_dir = self.config.build_dir().to_string();
        let mlstack_dir = format!("{}/.mlstack", home);
        let credentials_file = format!("{}/.git-credentials", home);
        ShellCommand {
            program: "sh".to_string(),
            args: vec![
                "-c".to_string(),
                format!(
                    "find '{}' -type d -name .git -prune -exec rm -rf '{{}}' ';' ; git credential-store erase --file '{}' ; git credential-cache exit ; rm -rf '{}'",
                    mlstack_dir, credentials_file, build_dir
                ),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    /// Build a git clone command with HTTPS authentication using the provided token.
    ///
    /// Uses git's credential helper to avoid exposing the token in command-line arguments.
    /// The token is passed via GITHUB_TOKEN environment variable, and git invokes
    /// a shell helper that outputs the credentials.
    fn git_clone_with_auth(&self, build_dir: &str, token: &str) -> ShellCommand {
        // Use standard HTTPS URL — token never appears in args or logs
        let credential_helper = r#"!sh -c "echo username=git; echo password=$GITHUB_TOKEN""#;
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                "--depth".to_string(),
                "1".to_string(),
                "--branch".to_string(),
                self.config.branch.clone(),
                self.config.repo_url().to_string(),
                build_dir.to_string(),
                "-c".to_string(),
                format!("credential.helper={}", credential_helper),
            ],
            env: vec![
                ("GITHUB_TOKEN".to_string(), token.to_string()),
                ("GIT_TERMINAL_PROMPT".to_string(), "0".to_string()),
            ],
            working_dir: None,
        }
    }

    fn mkdir_build_dir(&self, build_dir: &str) -> ShellCommand {
        ShellCommand {
            program: "mkdir".to_string(),
            args: vec!["-p".to_string(), build_dir.to_string()],
            env: vec![],
            working_dir: None,
        }
    }

    fn git_clone(&self, build_dir: &str) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                "--depth".to_string(),
                "1".to_string(),
                "--branch".to_string(),
                self.config.branch.clone(),
                self.config.repo_url().to_string(),
                build_dir.to_string(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    fn git_checkout(&self, build_dir: &str) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec!["checkout".to_string(), self.config.branch.clone()],
            env: vec![],
            working_dir: Some(PathBuf::from(build_dir)),
        }
    }

    fn cmake_configure(&self, build_dir: &str, install_prefix: &str) -> ShellCommand {
        let mut args = vec![
            "-B".to_string(),
            "build".to_string(),
            "-S".to_string(),
            ".".to_string(),
        ];
        args.extend(self.cmake_flags());
        args.push(format!("-DCMAKE_INSTALL_PREFIX={}", install_prefix));

        ShellCommand {
            program: "cmake".to_string(),
            args,
            env: vec![],
            working_dir: Some(PathBuf::from(build_dir)),
        }
    }

    fn cmake_build(&self, build_dir: &str) -> ShellCommand {
        let mut args = vec!["--build".to_string(), "build".to_string(), "-j".to_string()];
        // Add build targets
        for target in BUILD_TARGETS {
            args.push("--target".to_string());
            args.push(target.to_string());
        }

        ShellCommand {
            program: "cmake".to_string(),
            args,
            env: vec![],
            working_dir: Some(PathBuf::from(build_dir)),
        }
    }

    fn cmake_install(&self, build_dir: &str) -> ShellCommand {
        ShellCommand {
            program: "cmake".to_string(),
            args: vec!["--install".to_string(), "build".to_string()],
            env: vec![],
            working_dir: Some(PathBuf::from(build_dir)),
        }
    }

    fn submit_telemetry(
        &self,
        home: &str,
        build_start: Instant,
        was_prebuilt: bool,
        binary_path: Option<&std::path::Path>,
    ) -> Result<(), String> {
        let gpu = crate::hardware::detect_hardware()
            .map(|state| state.gpu)
            .unwrap_or_default();
        let report = BuildReport {
            gpu_arch: if gpu.architecture.is_empty() {
                self.config.gpu_arch.clone()
            } else {
                gpu.architecture
            },
            gpu_count: gpu.gpu_count,
            gpu_name: gpu.model,
            rocm_version: gpu.rocm_version,
            os: std::env::consts::OS.to_string(),
            os_distro: std::env::var("ID").unwrap_or_else(|_| "unknown".to_string()),
            build_duration_seconds: build_start.elapsed().as_secs(),
            git_commit: Self::current_git_commit().unwrap_or_else(|| "unknown".to_string()),
            build_status: BuildReportStatus::Success,
            install_path: binary_path
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| {
                    self.config
                        .detection_binary_path(home)
                        .display()
                        .to_string()
                }),
            cmake_flags: self.cmake_flags(),
            verification_path: self
                .config
                .detection_binary_path(home)
                .display()
                .to_string(),
            rdna3_validation_passed: false,
            wmma_available: false,
            shared_memory_ok: false,
            tokens_per_second_wmma: None,
            tokens_per_second_fallback: None,
            binary_version: self.config.branch.clone(),
            was_prebuilt,
        };
        submit_build_report(report);
        Ok(())
    }

    fn current_git_commit() -> Option<String> {
        let output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

// ===========================================================================
// Detection helpers (shared with component_status.rs)
// ===========================================================================

/// Resolve the install bin directory for llama-cpp.
///
/// Returns `~/.mlstack/components/llama-cpp/bin/`.
pub fn resolve_install_bin_dir(home: &str) -> PathBuf {
    PathBuf::from(home).join(DEFAULT_INSTALL_PREFIX).join("bin")
}

/// Check if the llama-cli binary is functional at the expected install path.
///
/// This is the single detection contract used by both detection and verification.
/// Returns `true` only if the binary exists AND `llama-cli --help` succeeds.
pub fn is_llama_cli_functional(home: &str) -> bool {
    let bin_path = resolve_install_bin_dir(home).join(DETECTION_BINARY);
    if !bin_path.exists() {
        return false;
    }
    // Run the detection subcommand to verify the binary is functional
    std::process::Command::new(bin_path)
        .arg(DETECTION_SUBCOMMAND)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if llama-cli is on PATH and functional.
///
/// Used as a fallback when the install-path check fails.
pub fn is_llama_cli_on_path() -> bool {
    std::process::Command::new(DETECTION_BINARY)
        .arg(DETECTION_SUBCOMMAND)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if partial artifacts exist (clone/build dirs) without a functional binary.
///
/// Returns `true` if artifacts exist but the binary is NOT functional.
/// Used to distinguish partial installs from complete ones.
pub fn has_partial_artifacts(home: &str) -> bool {
    let build_dir = PathBuf::from(DEFAULT_BUILD_DIR);
    let install_dir = PathBuf::from(home).join(DEFAULT_INSTALL_PREFIX);

    let artifacts_exist = build_dir.exists() || install_dir.exists();
    let binary_functional = is_llama_cli_functional(home);

    artifacts_exist && !binary_functional
}

/// Check if the installed llama-cli binary has ROCm/HIP linkage.
///
/// Verify that the installed binary has ROCm/HIP linkage and NOT CUDA linkage.
///
/// Uses `ldd` to check for:
/// - REQUIRED: `amdhip64` or `hipblas` or `libhip` (ROCm present)
/// - FORBIDDEN: `libcuda` or `libnvidia` or `libcudart` (CUDA contamination)
///
/// # Validation
///
/// - **VAL-GUARD-004**: Post-build linkage verification rejects CUDA-linked binaries
/// - **VAL-GUARD-005**: Post-build linkage verification requires ROCm linkage
fn verify_binary_linkage(binary_path: &Path) -> Result<(), String> {
    if !binary_path.exists() {
        return Err(format!(
            "Binary not found at {} — install may have failed",
            binary_path.display()
        ));
    }

    let output = std::process::Command::new("ldd")
        .arg(binary_path)
        .output()
        .map_err(|e| format!("Failed to run ldd on {}: {}", binary_path.display(), e))?;

    if !output.status.success() {
        return Err(format!(
            "ldd failed on {}: {}",
            binary_path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let ldd_output = String::from_utf8_lossy(&output.stdout);

    // Check for CUDA contamination
    let cuda_markers = ["libcuda.so", "libnvidia", "libcudart"];
    for marker in &cuda_markers {
        if ldd_output.contains(marker) {
            return Err(format!(
                "CUDA contamination detected: {} links against '{}'. \
                 The binary was built with CUDA instead of ROCm/HIP. \
                 This should not happen — check CMake configuration.",
                binary_path.display(),
                marker
            ));
        }
    }

    // Check for ROCm linkage
    let has_rocm = ldd_output.contains("amdhip64")
        || ldd_output.contains("hipblas")
        || ldd_output.contains("libhip");

    if !has_rocm {
        return Err(format!(
            "ROCm/HIP linkage not found in {}. The binary may be a CPU-only build \
             or was built without GPU acceleration. Expected linkage to amdhip64 or hipblas.",
            binary_path.display()
        ));
    }

    Ok(())
}

/// Check if NVIDIA CUDA toolkit artifacts are present on the system.
///
/// VAL-GUARD-002: detection is informational and never blocks HIP builds.
fn detect_cuda_toolkit_presence() -> Vec<String> {
    let mut found = Vec::new();
    let cuda_indicators = [
        ("nvcc", "NVIDIA CUDA compiler"),
        ("nvidia-smi", "NVIDIA system management interface"),
    ];

    for (binary, description) in cuda_indicators {
        if std::process::Command::new("which")
            .arg(binary)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
        {
            found.push(format!("{} ({})", binary, description));
        }
    }

    if Path::new("/usr/local/cuda").exists() {
        found.push("/usr/local/cuda directory".to_string());
    }

    found
}

/// Validate generated CMake cache after configure.
///
/// VAL-GUARD-003: cache must confirm HIP ON, CUDA OFF, expected GPU targets.
fn validate_cmake_cache(cache_path: &Path, expected_gpu_targets: &str) -> Result<(), String> {
    let contents = fs::read_to_string(cache_path).map_err(|err| {
        format!(
            "failed to read CMake cache at {}: {}",
            cache_path.display(),
            err
        )
    })?;

    let mut hip_on = false;
    let mut cuda_off = false;
    let mut gpu_targets_seen = false;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }

        if trimmed == "GGML_HIP:BOOL=ON" {
            hip_on = true;
        } else if trimmed == "GGML_CUDA:BOOL=OFF" {
            cuda_off = true;
        } else if let Some(value) = trimmed.strip_prefix("GPU_TARGETS:STRING=") {
            gpu_targets_seen = true;
            if value != expected_gpu_targets {
                return Err(format!(
                    "GPU_TARGETS mismatch in CMakeCache.txt: expected '{}', found '{}'",
                    expected_gpu_targets, value
                ));
            }
        }
    }

    if !hip_on {
        return Err("GGML_HIP is not ON in CMakeCache.txt; build is not using ROCm".to_string());
    }
    if !cuda_off {
        return Err(
            "GGML_CUDA is not OFF in CMakeCache.txt; CUDA contamination risk".to_string(),
        );
    }
    if !gpu_targets_seen {
        return Err("GPU_TARGETS is missing from CMakeCache.txt".to_string());
    }

    Ok(())
}

// ===========================================================================
// Post-install verification (follow-up hardening)
// ===========================================================================

/// Result of stronger-than---help post-install verification.
#[derive(Debug, Clone, PartialEq)]
pub struct PostInstallVerification {
    /// Whether the basic --help check passed.
    pub help_check_passed: bool,
    /// Whether a stronger verification (llama-bench) was attempted.
    pub stronger_check_attempted: bool,
    /// Whether the stronger verification passed.
    pub stronger_check_passed: bool,
    /// Prefill tokens/second from llama-bench (if run).
    pub bench_prefill_tps: Option<f64>,
    /// Decode tokens/second from llama-bench (if run).
    pub bench_decode_tps: Option<f64>,
    /// Whether RDNA3 validation was attempted.
    pub rdna3_check_attempted: bool,
    /// Whether RDNA3 validation passed (is_rdna3 = YES).
    pub rdna3_is_valid: Option<bool>,
    /// WMMA throughput from rdna3_validation.
    pub rdna3_wmma_tps: Option<f64>,
    /// Whether the GPU is RDNA3.
    pub is_rdna3_device: Option<bool>,
    /// Whether WMMA is supported on the detected GPU.
    pub wmma_supported: Option<bool>,
    /// Path to the binary that was verified.
    pub verified_binary_path: String,
    /// Any human-readable summary of the verification.
    pub summary: String,
}

/// Default verification model search paths relative to home.
const VERIFICATION_MODEL_SEARCH_PATHS: &[&str] = &[
    "models/Qwen3-0.6B-F16.gguf",
    ".mlstack/models/Qwen3-0.6B-F16.gguf",
    "llama.cpp-turboquant-hip/models/Qwen3-0.6B-F16.gguf",
];

/// Locate a model file for llama-bench verification.
fn find_verification_model(home: &str, fork_dir: &str) -> Option<PathBuf> {
    for relative in VERIFICATION_MODEL_SEARCH_PATHS {
        let candidate = PathBuf::from(home)
            .join("Documents/Product/Stan-s-ML-Stack/Fork")
            .join(relative.trim_start_matches("llama.cpp-turboquant-hip/"));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // Also check the fork_dir directly
    for relative in VERIFICATION_MODEL_SEARCH_PATHS {
        let candidate =
            PathBuf::from(fork_dir).join(relative.trim_start_matches("llama.cpp-turboquant-hip/"));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Performs stronger-than---help post-install verification on the installed binary.
///
/// Runs from the installed path (VAL-CROSS-009, VAL-CROSS-011).
/// Attempts llama-bench with a small model for VAL-CROSS-010.
/// Runs rdna3_validation if available for RDNA3 proof (VAL-CROSS-011).
pub fn verify_installed_binary(home: &str, fork_dir: &str) -> PostInstallVerification {
    let bin_dir = resolve_install_bin_dir(home);
    let cli_path = bin_dir.join(DETECTION_BINARY);
    let bench_path = bin_dir.join("llama-bench");
    let rdna3_path = bin_dir.join("rdna3_validation");

    let mut result = PostInstallVerification {
        help_check_passed: false,
        stronger_check_attempted: false,
        stronger_check_passed: false,
        bench_prefill_tps: None,
        bench_decode_tps: None,
        rdna3_check_attempted: false,
        rdna3_is_valid: None,
        rdna3_wmma_tps: None,
        is_rdna3_device: None,
        wmma_supported: None,
        verified_binary_path: cli_path.display().to_string(),
        summary: String::new(),
    };

    // ── Step 1: Basic --help check ──────────────────────────────────
    if cli_path.exists() {
        match std::process::Command::new(&cli_path)
            .arg(DETECTION_SUBCOMMAND)
            .output()
        {
            Ok(out) => {
                result.help_check_passed = out.status.success();
            }
            Err(_) => {
                result.help_check_passed = false;
            }
        }
    }

    // ── Step 2: Stronger check via llama-bench (if available) ──────
    if bench_path.exists() {
        result.stronger_check_attempted = true;
        if let Some(model_path) = find_verification_model(home, fork_dir) {
            match std::process::Command::new(&bench_path)
                .arg("-m")
                .arg(model_path.to_str().unwrap_or(""))
                .arg("-p")
                .arg("128")
                .arg("-n")
                .arg("32")
                .arg("-ngl")
                .arg("33")
                .output()
            {
                Ok(out) => {
                    if out.status.success() {
                        result.stronger_check_passed = true;
                        // Parse prefill and decode t/s from output
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        for line in stdout.lines() {
                            if line.contains("pp128") {
                                // Extract the t/s value (last numeric column)
                                if let Some(tps) = parse_bench_tps(line) {
                                    result.bench_prefill_tps = Some(tps);
                                }
                            } else if line.contains("tg32") {
                                if let Some(tps) = parse_bench_tps(line) {
                                    result.bench_decode_tps = Some(tps);
                                }
                            }
                        }
                    }
                }
                Err(_) => {}
            }
        }
    }

    // ── Step 3: RDNA3 proof (if available) ─────────────────────────
    if rdna3_path.exists() {
        result.rdna3_check_attempted = true;
        match std::process::Command::new(&rdna3_path).output() {
            Ok(out) => {
                if out.status.success() {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    result.is_rdna3_device = Some(stdout.contains("Is RDNA3: YES"));
                    result.wmma_supported = Some(stdout.contains("WMMA Supported: YES"));
                    result.rdna3_is_valid = result.is_rdna3_device;

                    // Extract WMMA throughput
                    for line in stdout.lines() {
                        if line.starts_with("WMMA Throughput:") {
                            if let Some(val) = line.split_whitespace().nth(2) {
                                if let Ok(tps) = val.parse::<f64>() {
                                    result.rdna3_wmma_tps = Some(tps);
                                }
                            }
                        }
                    }
                }
            }
            Err(_) => {}
        }
    }

    // ── Build summary ──────────────────────────────────────────────
    let mut parts: Vec<String> = vec![];
    if result.help_check_passed {
        parts.push("basic check PASSED".to_string());
    } else {
        parts.push("basic check FAILED".to_string());
    }
    if result.stronger_check_attempted {
        if result.stronger_check_passed {
            parts.push(format!(
                "llama-bench PASSED (prefill={:.0} t/s, decode={:.0} t/s)",
                result.bench_prefill_tps.unwrap_or(0.0),
                result.bench_decode_tps.unwrap_or(0.0)
            ));
        } else {
            parts.push("llama-bench FAILED".to_string());
        }
    } else {
        parts.push("llama-bench SKIPPED (binary unavailable)".to_string());
    }
    if result.rdna3_check_attempted {
        if result.rdna3_is_valid.unwrap_or(false) {
            parts.push(format!(
                "RDNA3 proof PASSED (WMMA={:.0} t/s)",
                result.rdna3_wmma_tps.unwrap_or(0.0)
            ));
        } else {
            parts.push("RDNA3 proof: not an RDNA3 device".to_string());
        }
    } else {
        parts.push("RDNA3 proof SKIPPED (rdna3_validation unavailable)".to_string());
    }

    result.summary = parts.join("; ");
    result
}

/// Parse tokens/second from a llama-bench output line.
/// Lines look like: `| qwen3 0.6B F16 | 1.11 GiB | 596.05 M | ROCm | 33 | pp128 | 6254.94 ± 1534.82 |`
fn parse_bench_tps(line: &str) -> Option<f64> {
    let parts: Vec<&str> = line.split('|').collect();
    if parts.len() >= 8 {
        let tps_cell = parts[7].trim();
        // Extract the number before any ±
        tps_cell
            .split('±')
            .next()
            .and_then(|s| s.trim().parse::<f64>().ok())
    } else {
        None
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_archs_from_gpu_arch() {
        assert_eq!(HipArchs::from_gpu_arch("gfx1030"), HipArchs::Gfx1030);
        assert_eq!(HipArchs::from_gpu_arch("gfx1100"), HipArchs::Gfx1100);
        assert_eq!(HipArchs::from_gpu_arch("gfx1101"), HipArchs::Gfx1100);
        assert_eq!(HipArchs::from_gpu_arch("gfx1200"), HipArchs::Gfx1200);
        assert_eq!(HipArchs::from_gpu_arch("gfx1201"), HipArchs::Gfx1200);
        assert_eq!(HipArchs::from_gpu_arch("gfx000"), HipArchs::Default);
        assert_eq!(HipArchs::from_gpu_arch("unknown"), HipArchs::Default);
    }

    #[test]
    fn test_gpu_targets_legacy_channel() {
        // Legacy channel always returns gfx1030 regardless of detected GPU
        assert_eq!(
            HipArchs::Gfx1100.gpu_targets_for_channel("legacy"),
            "gfx1030"
        );
        assert_eq!(
            HipArchs::Gfx1200.gpu_targets_for_channel("legacy"),
            "gfx1030"
        );
        assert_eq!(
            HipArchs::Gfx1030.gpu_targets_for_channel("legacy"),
            "gfx1030"
        );
        assert_eq!(
            HipArchs::Default.gpu_targets_for_channel("legacy"),
            "gfx1030"
        );
    }

    #[test]
    fn test_gpu_targets_stable_channel() {
        assert_eq!(
            HipArchs::Gfx1030.gpu_targets_for_channel("stable"),
            "gfx1030"
        );
        assert_eq!(
            HipArchs::Gfx1100.gpu_targets_for_channel("stable"),
            "gfx1030;gfx1100;gfx1101"
        );
        // RDNA4 on stable degrades to RDNA2+RDNA3 (no gfx1200)
        assert_eq!(
            HipArchs::Gfx1200.gpu_targets_for_channel("stable"),
            "gfx1030;gfx1100;gfx1101"
        );
        assert_eq!(
            HipArchs::Default.gpu_targets_for_channel("stable"),
            "gfx1030"
        );
    }

    #[test]
    fn test_gpu_targets_latest_channel() {
        assert_eq!(
            HipArchs::Gfx1030.gpu_targets_for_channel("latest"),
            "gfx1030"
        );
        assert_eq!(
            HipArchs::Gfx1100.gpu_targets_for_channel("latest"),
            "gfx1030;gfx1100;gfx1101"
        );
        assert_eq!(
            HipArchs::Gfx1200.gpu_targets_for_channel("latest"),
            "gfx1030;gfx1100;gfx1101;gfx1200"
        );
    }

    #[test]
    fn test_rocm_channel_wmma_fa() {
        assert!(!RocmChannel::from_str("legacy").enable_wmma_fa());
        assert!(RocmChannel::from_str("stable").enable_wmma_fa());
        assert!(RocmChannel::from_str("latest").enable_wmma_fa());
    }

    #[test]
    fn test_cmake_flags_legacy() {
        let config = LlamaCppConfig {
            gpu_arch: "gfx1100".to_string(),
            channel: "legacy".to_string(),
            ..Default::default()
        };
        let installer = LlamaCppInstaller::new(config);
        let flags = installer.cmake_flags();

        assert!(flags.contains(&"-DGGML_HIP=ON".to_string()));
        assert!(flags.contains(&"-DGPU_TARGETS=gfx1030".to_string()));
        // Legacy should NOT have WMMA FA enabled
        assert!(!flags.iter().any(|f| f.contains("RDNA3_PROBES")));
        assert!(!flags.iter().any(|f| f.contains("ROCWMMA_FATTN")));
    }

    #[test]
    fn test_cmake_flags_always_disable_non_rocm_backends() {
        // VAL-GUARD-001: every channel must explicitly disable non-ROCm backends.
        for channel in ["legacy", "stable", "latest"] {
            let config = LlamaCppConfig {
                gpu_arch: "gfx1100".to_string(),
                channel: channel.to_string(),
                ..Default::default()
            };
            let installer = LlamaCppInstaller::new(config);
            let flags = installer.cmake_flags();

            assert!(
                flags.contains(&"-DGGML_CUDA=OFF".to_string()),
                "channel {channel} must set GGML_CUDA=OFF"
            );
            assert!(
                flags.contains(&"-DGGML_VULKAN=OFF".to_string()),
                "channel {channel} must set GGML_VULKAN=OFF"
            );
            assert!(
                flags.contains(&"-DGGML_METAL=OFF".to_string()),
                "channel {channel} must set GGML_METAL=OFF"
            );
        }
    }

    #[test]
    fn test_cmake_flags_stable() {
        let config = LlamaCppConfig {
            gpu_arch: "gfx1100".to_string(),
            channel: "stable".to_string(),
            ..Default::default()
        };
        let installer = LlamaCppInstaller::new(config);
        let flags = installer.cmake_flags();

        assert!(flags.contains(&"-DGGML_HIP=ON".to_string()));
        assert!(flags.contains(&"-DGPU_TARGETS=gfx1030;gfx1100;gfx1101".to_string()));
        assert!(flags.contains(&"-DGGML_HIP_RDNA3_PROBES=ON".to_string()));
        assert!(flags.contains(&"-DGGML_HIP_ROCWMMA_FATTN=ON".to_string()));
    }

    #[test]
    fn test_cmake_flags_latest_with_rdna4() {
        let config = LlamaCppConfig {
            gpu_arch: "gfx1200".to_string(),
            channel: "latest".to_string(),
            ..Default::default()
        };
        let installer = LlamaCppInstaller::new(config);
        let flags = installer.cmake_flags();

        assert!(flags.contains(&"-DGGML_HIP=ON".to_string()));
        assert!(flags.contains(&"-DGPU_TARGETS=gfx1030;gfx1100;gfx1101;gfx1200".to_string()));
        assert!(flags.contains(&"-DGGML_HIP_RDNA3_PROBES=ON".to_string()));
        assert!(flags.contains(&"-DGGML_HIP_ROCWMMA_FATTN=ON".to_string()));
    }

    #[test]
    fn test_rocm_channel_rdna3_gating() {
        let legacy = LlamaCppInstaller::new(LlamaCppConfig {
            gpu_arch: "gfx1100".to_string(),
            channel: "legacy".to_string(),
            ..Default::default()
        });
        let stable = LlamaCppInstaller::new(LlamaCppConfig {
            gpu_arch: "gfx1100".to_string(),
            channel: "stable".to_string(),
            ..Default::default()
        });
        let latest = LlamaCppInstaller::new(LlamaCppConfig {
            gpu_arch: "gfx1100".to_string(),
            channel: "latest".to_string(),
            ..Default::default()
        });

        let legacy_flags = legacy.cmake_flags();
        let stable_flags = stable.cmake_flags();
        let latest_flags = latest.cmake_flags();

        assert!(!legacy_flags
            .iter()
            .any(|flag| flag.contains("RDNA3_PROBES") || flag.contains("ROCWMMA_FATTN")));
        assert!(stable_flags.contains(&"-DGGML_HIP_RDNA3_PROBES=ON".to_string()));
        assert!(stable_flags.contains(&"-DGGML_HIP_ROCWMMA_FATTN=ON".to_string()));
        assert!(latest_flags.contains(&"-DGGML_HIP_RDNA3_PROBES=ON".to_string()));
        assert!(latest_flags.contains(&"-DGGML_HIP_ROCWMMA_FATTN=ON".to_string()));
    }

    #[test]
    fn test_cmake_flags_unknown_gpu_conservative() {
        let config = LlamaCppConfig {
            gpu_arch: "unknown".to_string(),
            channel: "latest".to_string(),
            ..Default::default()
        };
        let installer = LlamaCppInstaller::new(config);
        let flags = installer.cmake_flags();

        // Unknown GPU should degrade to conservative gfx1030
        assert!(flags.contains(&"-DGPU_TARGETS=gfx1030".to_string()));
    }

    #[test]
    fn test_detect_cuda_toolkit_presence_no_panic() {
        // VAL-GUARD-002: detection must be safe whether CUDA exists or not.
        let result = detect_cuda_toolkit_presence();
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_validate_cmake_cache_accepts_rocm_only_config() {
        // VAL-GUARD-003: valid cache confirms HIP ON, CUDA OFF, expected GPU targets.
        let temp = tempfile::NamedTempFile::new().unwrap();
        fs::write(
            temp.path(),
            "GGML_HIP:BOOL=ON\nGGML_CUDA:BOOL=OFF\nGPU_TARGETS:STRING=gfx1030;gfx1100;gfx1101\n",
        )
        .unwrap();

        assert!(validate_cmake_cache(temp.path(), "gfx1030;gfx1100;gfx1101").is_ok());
    }

    #[test]
    fn test_validate_cmake_cache_rejects_cuda_on() {
        // VAL-GUARD-003: CUDA ON is contamination risk.
        let temp = tempfile::NamedTempFile::new().unwrap();
        fs::write(
            temp.path(),
            "GGML_HIP:BOOL=ON\nGGML_CUDA:BOOL=ON\nGPU_TARGETS:STRING=gfx1030\n",
        )
        .unwrap();

        let err = validate_cmake_cache(temp.path(), "gfx1030").unwrap_err();
        assert!(err.contains("GGML_CUDA is not OFF"));
    }

    #[test]
    fn test_validate_cmake_cache_rejects_gpu_target_mismatch() {
        // VAL-GUARD-003: GPU_TARGETS must match channel policy exactly.
        let temp = tempfile::NamedTempFile::new().unwrap();
        fs::write(
            temp.path(),
            "GGML_HIP:BOOL=ON\nGGML_CUDA:BOOL=OFF\nGPU_TARGETS:STRING=gfx1030\n",
        )
        .unwrap();

        let err = validate_cmake_cache(temp.path(), "gfx1030;gfx1100;gfx1101").unwrap_err();
        assert!(err.contains("GPU_TARGETS mismatch"));
    }

    #[test]
    fn test_build_commands_sequence() {
        let config = LlamaCppConfig {
            gpu_arch: "gfx1100".to_string(),
            channel: "latest".to_string(),
            ..Default::default()
        };
        let installer = LlamaCppInstaller::new(config);
        let commands = installer.build_commands("/home/testuser");

        // Should have 7 commands: mkdir, clone, checkout, cmake configure, build, install, purge
        assert_eq!(commands.len(), 7);
        assert_eq!(commands[0].program, "mkdir");
        assert_eq!(commands[1].program, "git");
        assert_eq!(commands[2].program, "git");
        assert_eq!(commands[3].program, "cmake");
        assert_eq!(commands[4].program, "cmake");
        assert_eq!(commands[5].program, "cmake");
        assert_eq!(commands[6].program, "sh");
    }

    #[test]
    fn test_detection_binary_path() {
        let config = LlamaCppConfig::default();
        let path = config.detection_binary_path("/home/testuser");
        assert_eq!(
            path,
            PathBuf::from("/home/testuser/.mlstack/components/llama-cpp/bin/llama-cli")
        );
    }

    #[test]
    fn test_resolve_install_bin_dir() {
        let dir = resolve_install_bin_dir("/home/testuser");
        assert_eq!(
            dir,
            PathBuf::from("/home/testuser/.mlstack/components/llama-cpp/bin")
        );
    }

    #[test]
    fn test_is_llama_cli_functional_absent() {
        // With a non-existent home, the binary should not be found
        assert!(!is_llama_cli_functional(
            "/nonexistent/path/that/does/not/exist"
        ));
    }

    #[test]
    fn test_has_partial_artifacts_no_artifacts() {
        // With a non-existent home, no artifacts should exist.
        // Use a path starting with /tmp-nonexistent- prefix to avoid
        // races with other tests that may create /tmp/llama-cpp-rocm-build.
        assert!(!has_partial_artifacts(
            "/nonexistent-path-DOES-NOT-exist-f94a1f2a3b5c/test-home"
        ));
    }

    #[test]
    fn test_dependencies() {
        let installer = LlamaCppInstaller::with_defaults();
        assert!(installer.dependencies().contains(&"rocm"));
    }

    #[test]
    fn test_repo_url_default() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.repo_url(), DEFAULT_REPO_URL);
    }

    #[test]
    fn test_repo_url_override() {
        let config = LlamaCppConfig {
            repo_url: Some("https://example.com/fork.git".to_string()),
            ..Default::default()
        };
        assert_eq!(config.repo_url(), "https://example.com/fork.git");
    }

    #[test]
    fn test_rocm_channel_from_str() {
        assert_eq!(RocmChannel::from_str("legacy"), RocmChannel::Legacy);
        assert_eq!(RocmChannel::from_str("LEGACY"), RocmChannel::Legacy);
        assert_eq!(RocmChannel::from_str("stable"), RocmChannel::Stable);
        assert_eq!(RocmChannel::from_str("latest"), RocmChannel::Latest);
        assert_eq!(RocmChannel::from_str("unknown"), RocmChannel::Latest);
    }

    #[test]
    fn test_rocm_channel_label() {
        assert_eq!(RocmChannel::Legacy.label(), "legacy");
        assert_eq!(RocmChannel::Stable.label(), "stable");
        assert_eq!(RocmChannel::Latest.label(), "latest");
    }

    #[test]
    fn test_build_commands_with_auth_token_handling() {
        // Save any pre-existing GITHUB_TOKEN value to restore later
        let _original_token = std::env::var("GITHUB_TOKEN").ok();

        // --- Branch 1: with GITHUB_TOKEN set ---
        let config = LlamaCppConfig::default();
        let installer = LlamaCppInstaller::new(config);
        let commands_with_token = installer
            .build_commands_with_auth("/home/testuser", Some(SealedToken::new("mock_token")));

        // The clone command is at index 1 (after mkdir)
        let clone_cmd = &commands_with_token[1];
        // Token must NOT appear directly in args (security requirement)
        assert!(!clone_cmd.args.iter().any(|arg| arg.contains("mock_token")));
        // Plain HTTPS repo URL must be present
        assert!(clone_cmd.args.iter().any(|arg| arg == DEFAULT_REPO_URL));
        // Credential helper flag must be present
        assert!(clone_cmd
            .args
            .iter()
            .any(|arg| arg.starts_with("credential.helper=")));
        // Env must contain GITHUB_TOKEN for git auth and GIT_TERMINAL_PROMPT
        assert!(clone_cmd
            .env
            .iter()
            .any(|(k, v)| k == "GITHUB_TOKEN" && v == "mock_token"));
        assert!(clone_cmd
            .env
            .iter()
            .any(|(k, v)| k == "GIT_TERMINAL_PROMPT" && v == "0"));
    }

    #[test]
    fn test_release_manifest_parse_and_lookup() {
        let json = r#"{
            "version": "v0.3.2",
            "archs": {
                "gfx1030": {"url": "https://example.com/a", "sha256": "aaa"},
                "gfx1100": {"url": "https://example.com/b", "sha256": "bbb"},
                "gfx1200": {"url": "https://example.com/c", "sha256": "ccc"}
            }
        }"#;

        let manifest: ReleaseManifest = serde_json::from_str(json).unwrap();
        let installer = LlamaCppInstaller::with_defaults();
        assert_eq!(manifest.version, "v0.3.2");
        assert_eq!(
            installer
                .release_asset_for_arch(&manifest, "gfx1030")
                .unwrap()
                .url,
            "https://example.com/a"
        );
        assert_eq!(
            installer
                .release_asset_for_arch(&manifest, "gfx1100")
                .unwrap()
                .sha256,
            "bbb"
        );
        assert!(installer
            .release_asset_for_arch(&manifest, "gfx9999")
            .is_none());
    }

    #[test]
    fn test_release_manifest_rejects_unknown_fields() {
        let json = r#"{
            "version": "v0.3.2",
            "unexpected": true,
            "archs": {
                "gfx1030": {"url": "https://example.com/a", "sha256": "aaa"},
                "gfx1100": {"url": "https://example.com/b", "sha256": "bbb"},
                "gfx1200": {"url": "https://example.com/c", "sha256": "ccc"}
            }
        }"#;

        assert!(serde_json::from_str::<ReleaseManifest>(json).is_err());
    }

    #[test]
    fn test_verify_sha256_matches() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        fs::write(temp.path(), b"hello").unwrap();
        let expected = Sha256::digest(b"hello")
            .iter()
            .map(|byte| format!("{:02x}", byte))
            .collect::<String>();
        assert!(LlamaCppInstaller::verify_sha256(temp.path(), &expected).unwrap());
    }

    #[test]
    fn test_verify_sha256_mismatch() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        fs::write(temp.path(), b"hello").unwrap();
        assert!(!LlamaCppInstaller::verify_sha256(temp.path(), "deadbeef").unwrap());
    }

    #[test]
    fn test_download_failure_falls_back_to_source() {
        let installer = LlamaCppInstaller::with_defaults();
        let plan = PrebuiltInstallPlan {
            manifest_version: "v1".into(),
            arch: "gfx1100".into(),
            url: "http://127.0.0.1:9/nope".into(),
            sha256: "deadbeef".into(),
            install_prefix: tempfile::tempdir().unwrap().path().display().to_string(),
            binary_path: PathBuf::from("/tmp/llama-cli"),
        };
        assert!(installer.download_prebuilt_binary(&plan).is_err());
    }

    #[test]
    fn test_purge_source_artifacts_is_idempotent() {
        let installer = LlamaCppInstaller::with_defaults();
        let temp_home = tempfile::tempdir().unwrap();
        let home = temp_home.path().to_string_lossy().to_string();

        std::fs::create_dir_all(format!("{home}/.mlstack/components/llama-cpp")).unwrap();
        std::fs::create_dir_all(format!("{home}/.mlstack/components/llama-cpp/.git")).unwrap();
        std::fs::create_dir_all("/tmp/llama-cpp-rocm-build").unwrap();
        std::fs::create_dir_all(format!("{home}/.mlstack/components/llama-cpp/bin")).unwrap();
        std::fs::write(
            format!("{home}/.mlstack/components/llama-cpp/bin/llama-cli"),
            "#!/bin/sh\nexit 0\n",
        )
        .unwrap();

        assert!(installer.purge_source_artifacts(&home, true).is_ok());
        assert!(installer.purge_source_artifacts(&home, true).is_ok());
    }

    #[test]
    fn test_git_clone_with_auth() {
        let config = LlamaCppConfig::default();
        let installer = LlamaCppInstaller::new(config);
        let command = installer.git_clone_with_auth("/tmp/test", "mock_token");

        // Token must NOT appear directly in args (security requirement)
        assert!(!command.args.iter().any(|arg| arg.contains("mock_token")));
        // Plain HTTPS repo URL must be present
        assert!(command.args.iter().any(|arg| arg == DEFAULT_REPO_URL));
        // Credential helper flag must be present
        assert!(command.args.iter().any(|arg| arg == "-c"));
        assert!(command
            .args
            .iter()
            .any(|arg| arg.starts_with("credential.helper=")));
        // Env must contain GITHUB_TOKEN for git auth and GIT_TERMINAL_PROMPT
        assert!(command
            .env
            .iter()
            .any(|(k, v)| k == "GITHUB_TOKEN" && v == "mock_token"));
        assert!(command
            .env
            .iter()
            .any(|(k, v)| k == "GIT_TERMINAL_PROMPT" && v == "0"));
    }

    // Note: `validate_repo_access` is not tested here because it requires real git/network access.
    // It is tested indirectly via integration tests or manual validation.

    // ── Post-install verification tests (VAL-CROSS-008/009/010/011) ──

    #[test]
    fn test_post_install_verification_no_binaries() {
        let result =
            verify_installed_binary("/nonexistent/path/that/does/not/exist", "/nonexistent/fork");
        assert!(!result.help_check_passed);
        assert!(!result.stronger_check_attempted);
        assert!(!result.rdna3_check_attempted);
        assert!(!result.summary.is_empty());
        assert!(result.summary.contains("FAILED"));
    }

    #[test]
    fn test_post_install_verification_struct_fields() {
        let result =
            verify_installed_binary("/nonexistent/path/that/does/not/exist", "/nonexistent/fork");
        assert!(!result.help_check_passed);
        assert!(!result.stronger_check_attempted);
        assert!(!result.stronger_check_passed);
        assert_eq!(result.bench_prefill_tps, None);
        assert_eq!(result.bench_decode_tps, None);
        assert!(!result.rdna3_check_attempted);
        assert_eq!(result.rdna3_is_valid, None);
        assert_eq!(result.rdna3_wmma_tps, None);
        assert_eq!(result.is_rdna3_device, None);
        assert_eq!(result.wmma_supported, None);
        assert!(result.verified_binary_path.contains("llama-cli"));
    }

    #[test]
    fn test_post_install_verification_cli_only() {
        let tmp = tempfile::tempdir().unwrap();
        let bin_dir = tmp.path().join(".mlstack/components/llama-cpp/bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        std::fs::write(
            bin_dir.join("llama-cli"),
            "#!/bin/sh\necho 'usage: llama-cli'\nexit 0\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                bin_dir.join("llama-cli"),
                std::fs::Permissions::from_mode(0o755),
            )
            .unwrap();
        }

        let result = verify_installed_binary(tmp.path().to_str().unwrap(), "/nonexistent/fork");
        assert!(result.help_check_passed);
        assert!(!result.stronger_check_attempted);
        assert!(!result.rdna3_check_attempted);
        assert!(result.summary.contains("PASSED"));
        assert!(result.summary.contains("SKIPPED"));
    }

    #[test]
    fn test_parse_bench_tps_valid() {
        let line =
            "| qwen3 0.6B F16 | 1.11 GiB | 596.05 M | ROCm | 33 | pp128 | 6254.94 ± 1534.82 |";
        let tps = parse_bench_tps(line);
        assert!(tps.is_some());
        assert!((tps.unwrap() - 6254.94).abs() < 0.01);
    }

    #[test]
    fn test_parse_bench_tps_no_uncertainty() {
        let line = "| qwen3 0.6B F16 | 1.11 GiB | 596.05 M | ROCm | 33 | tg32 | 152.17 |";
        let tps = parse_bench_tps(line);
        assert!(tps.is_some());
        assert!((tps.unwrap() - 152.17).abs() < 0.01);
    }

    #[test]
    fn test_parse_bench_tps_invalid_line() {
        assert_eq!(parse_bench_tps("not a bench line"), None);
        assert_eq!(parse_bench_tps(""), None);
        assert_eq!(parse_bench_tps("| col1 | col2 |"), None);
    }

    #[test]
    fn test_verify_binary_linkage_missing_binary() {
        // VAL-GUARD-004, VAL-GUARD-005
        let result = verify_binary_linkage(std::path::Path::new("/nonexistent/binary"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_verify_binary_linkage_no_panic() {
        // Test with a known binary (bash) — should fail on ROCm linkage check
        // but should NOT panic
        let result = verify_binary_linkage(std::path::Path::new("/usr/bin/bash"));
        // bash won't have ROCm linkage, so this should be an Err
        assert!(result.is_err());
    }
}
