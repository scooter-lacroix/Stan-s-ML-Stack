# PR: ROCm Reinstall, Multi-GPU Runtime, and Benchmark Stabilization

## Proposed Title

`stabilization: harden ROCm reinstall flow, benchmark runtime, and vLLM/DeepSpeed/Megatron reliability`

## Summary

This PR delivers a full ROCm and benchmark stabilization wave across installer scripts, runtime environment setup, and Rusty-Stack benchmark UX/reporting.

Primary outcomes:

- ROCm force reinstall now follows a strict purge/reboot/resume/reboot lifecycle.
- Cross-distro package handling was hardened for Debian/Ubuntu, Arch, Fedora/RHEL, and openSUSE.
- Mixed iGPU+dGPU systems now consistently filter iGPU indices from visible-device runtime exports.
- vLLM runtime dependency reconciliation is applied in installer and benchmark preflight paths.
- DeepSpeed and Megatron benchmark paths were stabilized and integrated.
- HTML benchmark export now includes actionable charts/tables and user-facing export notifications.

## Problem Statement

Before this PR, users encountered:

- ROCm reinstall flows that exited early and did not perform complete purge/reinstall cycles.
- Repeated vLLM benchmark failures due to missing runtime dependencies and runtime init issues.
- DeepSpeed benchmark empty/no-data or runtime failures.
- Mixed GPU visibility where integrated GPUs leaked into runtime device variables.
- HTML export lacking enough context (axes/tables/notification) for report usability.

## Scope

### ROCm Installer and Reinstall Flow

- `scripts/install_rocm.sh`
  - Added forced purge engines for package-manager families (`apt/dpkg`, `dnf/yum`, `zypper`, `pacman`).
  - Added force-reinstall state handling and reboot choreography.
  - Added resume-after-reboot launcher/autostart support.
  - Added second reboot finalization after successful reinstall.
  - Hardened Arch ROCm flow:
    - repo package install through `pacman`,
    - AUR package install through non-root helper,
    - AUR availability checks,
    - user sudo ticket prime + keepalive.

### Runtime Environment and iGPU Filtering

- `scripts/setup_permanent_rocm_env.sh`
- `scripts/lib/benchmark_common.sh`
- `rusty-stack/src/benchmarks/mod.rs`
- `rusty-stack/src/widgets/benchmarks_page.rs`
  - Strengthened integrated GPU detection heuristics.
  - Enforced discrete-only visible-device exports in persistent and benchmark runtime setup.
  - Added fish-compatible environment generation and startup integration.
  - Added managed Triton cache environment setup for writable cache paths.

### vLLM Installer and Benchmark Reliability

- `scripts/install_vllm_multi.sh`
- `scripts/run_vllm_benchmarks.sh`
- `scripts/lib/benchmark_common.sh`
  - Normalized ROCm target-device environment.
  - Added/expanded missing-module reconciliation for common vLLM dependency gaps.
  - Added runtime preflight and repair behavior before benchmark execution.
  - Improved benchmark logging context (visible devices, target device, cache path, tiny model selection).

### DeepSpeed and Megatron

- `scripts/run_deepspeed_benchmarks.sh`
- `scripts/install_megatron.sh`
- `scripts/run_megatron_benchmarks.sh` (new)
- `scripts/run_all_benchmarks_suite.sh`
  - DeepSpeed benchmark preflight and output handling improved.
  - Megatron install path hardened with runtime dependency reconciliation and verification handling.
  - Added dedicated Megatron benchmark runner and full-suite integration.

### Rusty-Stack Benchmark UX and Reporting

- `rusty-stack/src/app.rs`
- `rusty-stack/src/widgets/benchmarks_page.rs`
- `rusty-stack/src/benchmark_logs.rs` (new)
- `rusty-stack/src/installer.rs`
  - Added robust benchmark JSON extraction/parsing from mixed logs.
  - Added benchmark HTML export feedback in UI.
  - Expanded exported HTML report with labeled axes, plotted data points, and summary/detail tables.

## Documentation Updates in This PR

- `CHANGELOG.md` (`Unreleased` -> `Platform Stabilization (2026-02)`)
- `docs/core/rocm_installation_guide.md`
- `docs/extensions/vllm_guide.md`
- `docs/rusty_stack_guide.md`
- `docs/guides/troubleshooting_guide.md`
- `docs/INSTALLER_STATUS.md`

## Validation Evidence

Executed:

- `cargo check --manifest-path rusty-stack/Cargo.toml -q`
- `bash -n scripts/install_rocm.sh scripts/setup_permanent_rocm_env.sh scripts/lib/benchmark_common.sh scripts/install_vllm_multi.sh scripts/run_vllm_benchmarks.sh scripts/run_deepspeed_benchmarks.sh scripts/run_megatron_benchmarks.sh`

Observed benchmark evidence:

- vLLM benchmark success log with throughput metrics and discrete-only visible devices (example payload fields: `success=true`, `throughput_tokens_per_sec`, `visible_devices=0,1`).

## Risk Assessment

- Medium: ROCm reinstall path changes are broad and system-level.
- Low-to-medium: runtime env filtering may affect edge-case mixed-GPU enumeration.
- Low: HTML/reporting and parser changes are additive and isolated.

## Rollback Plan

1. Revert this PR commit range.
2. Re-run prior installer flow without `--force`.
3. Regenerate `~/.mlstack_env` if necessary.
4. Revalidate with baseline benchmark commands.

## Merge Checklist

- [ ] CI/build checks pass.
- [ ] Maintainer spot-tests ROCm force reinstall on one distro per package family.
- [ ] Maintainer confirms benchmark HTML export content and UI notification behavior.
- [ ] Maintainer verifies vLLM/DeepSpeed/Megatron benchmark commands produce parseable outputs.
- [ ] Docs reviewed for consistency with implemented behavior.
