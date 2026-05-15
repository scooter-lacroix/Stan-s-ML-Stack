use super::{log_warn, SealedToken};
use crate::state::GPUInfo;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BuildReportStatus {
    Success,
    Failure,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BuildReport {
    pub gpu_arch: String,
    pub gpu_count: usize,
    pub gpu_name: String,
    pub rocm_version: String,
    pub os: String,
    pub os_distro: String,
    pub build_duration_seconds: u64,
    pub git_commit: String,
    pub build_status: BuildReportStatus,
    pub install_path: String,
    pub cmake_flags: Vec<String>,
    pub verification_path: String,
    pub rdna3_validation_passed: bool,
    pub wmma_available: bool,
    pub shared_memory_ok: bool,
    pub tokens_per_second_wmma: Option<f32>,
    pub tokens_per_second_fallback: Option<f32>,
    pub binary_version: String,
    pub was_prebuilt: bool,
}

impl BuildReport {
    pub fn from_hardware(
        gpu: &GPUInfo,
        os: impl Into<String>,
        os_distro: impl Into<String>,
        build_duration: Duration,
        git_commit: impl Into<String>,
        install_path: impl Into<String>,
        cmake_flags: Vec<String>,
        verification_path: impl Into<String>,
        binary_version: impl Into<String>,
        was_prebuilt: bool,
    ) -> Self {
        Self {
            gpu_arch: gpu.architecture.clone(),
            gpu_count: gpu.gpu_count,
            gpu_name: gpu.model.clone(),
            rocm_version: gpu.rocm_version.clone(),
            os: os.into(),
            os_distro: os_distro.into(),
            build_duration_seconds: build_duration.as_secs(),
            git_commit: git_commit.into(),
            build_status: BuildReportStatus::Success,
            install_path: install_path.into(),
            cmake_flags,
            verification_path: verification_path.into(),
            rdna3_validation_passed: false,
            wmma_available: false,
            shared_memory_ok: false,
            tokens_per_second_wmma: None,
            tokens_per_second_fallback: None,
            binary_version: binary_version.into(),
            was_prebuilt,
        }
    }
}

pub fn submit_build_report(report: BuildReport) {
    let mut token = SealedToken::from_env();
    let payload = build_report_payload(&report);

    let response = ureq::post(submission_endpoint())
        .header("Authorization", format!("Bearer {}", token.as_str()))
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .header("User-Agent", "rusty-stack")
        .send_json(payload);

    if let Err(err) = response {
        log_warn(&format!("telemetry dispatch failed: {}", err));
    }

    token.purge();
}

pub fn build_report_from_state(
    gpu: &GPUInfo,
    os: impl Into<String>,
    os_distro: impl Into<String>,
    build_duration: Duration,
    git_commit: impl Into<String>,
    install_path: impl Into<String>,
    cmake_flags: Vec<String>,
    verification_path: impl Into<String>,
    binary_version: impl Into<String>,
    was_prebuilt: bool,
) -> BuildReport {
    BuildReport::from_hardware(
        gpu,
        os,
        os_distro,
        build_duration,
        git_commit,
        install_path,
        cmake_flags,
        verification_path,
        binary_version,
        was_prebuilt,
    )
}

pub fn build_report_payload(report: &BuildReport) -> serde_json::Value {
    serde_json::json!({
        "event_type": "build-validation-report",
        "client_payload": report,
    })
}

pub fn submission_endpoint() -> &'static str {
    "https://api.github.com/repos/scooter-lacroix/llama.cpp-turboquant-hip/dispatches"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn build_report_serializes_all_fields() {
        let gpu = GPUInfo {
            architecture: "gfx1100".into(),
            gpu_count: 2,
            model: "AMD Radeon RX 7900 XTX".into(),
            rocm_version: "7.2.1".into(),
            ..Default::default()
        };
        let report = BuildReport::from_hardware(
            &gpu,
            "linux",
            "ubuntu",
            Duration::from_secs(42),
            "abc123",
            "/tmp/install",
            vec!["-DGGML_HIP=ON".into()],
            "/tmp/verify",
            "v0.3.2",
            true,
        );

        let json = serde_json::to_value(report).unwrap();
        for key in [
            "gpu_arch",
            "gpu_count",
            "gpu_name",
            "rocm_version",
            "os",
            "os_distro",
            "build_duration_seconds",
            "git_commit",
            "build_status",
            "install_path",
            "cmake_flags",
            "verification_path",
            "rdna3_validation_passed",
            "wmma_available",
            "shared_memory_ok",
            "tokens_per_second_wmma",
            "tokens_per_second_fallback",
            "binary_version",
            "was_prebuilt",
        ] {
            assert!(json.get(key).is_some(), "missing key {key}");
        }
    }

    #[test]
    fn submission_payload_matches_repository_dispatch_format() {
        let gpu = GPUInfo {
            architecture: "gfx1100".into(),
            gpu_count: 1,
            model: "AMD Radeon RX 7900 XTX".into(),
            rocm_version: "7.2.1".into(),
            ..Default::default()
        };
        let report = BuildReport::from_hardware(
            &gpu,
            "linux",
            "ubuntu",
            Duration::from_secs(12),
            "commit123",
            "/opt/mlstack",
            vec!["-DGGML_HIP=ON".into(), "-DGPU_TARGETS=gfx1100".into()],
            "/tmp/verify",
            "v0.3.2",
            false,
        );

        let payload = build_report_payload(&report);
        assert_eq!(
            payload["event_type"],
            serde_json::Value::String("build-validation-report".into())
        );
        assert_eq!(payload["client_payload"]["gpu_arch"], "gfx1100");
        assert_eq!(payload["client_payload"]["build_duration_seconds"], 12);
        assert!(payload["client_payload"]["cmake_flags"].is_array());
    }
}
