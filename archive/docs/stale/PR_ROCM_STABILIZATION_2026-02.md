# PR: ROCm Reinstall, Multi-GPU Runtime, and Benchmark Stabilization

## Proposed Title

`stabilization: pre-Anagami ROCm/runtime/benchmark hardening (no release tag)`

## **Release Positioning**

- <u>**No new release is cut by this PR.**</u>
- This is the **Sotapanna (0.1.4) -> Anagami transition stabilization update** tracked under `Unreleased`.
- *Anagami release gate*: installer/backend migration to Rust as a single package publishable to crates.io.

## **Executive Summary**

This PR consolidates a full reliability pass over ROCm installation/reinstallation, runtime environment generation, and benchmark execution/reporting.

**Key outcomes**:

- **ROCm force reinstall** now follows strict purge/reboot/resume/reboot sequencing.
- **Cross-distro reliability** improved for Debian/Ubuntu, Arch/CachyOS, Fedora/RHEL, and openSUSE.
- **Mixed GPU hygiene** improved by filtering iGPU visibility from runtime device exports.
- **vLLM/DeepSpeed/Megatron** benchmark/install reliability was hardened with dependency and runtime preflight work.
- **Benchmark UX/reporting** now provides export notifications plus a comprehensive HTML report.

## **Problem Statement**

Before this PR, the stack could fail or degrade in several recurring ways:

- force reinstall flows that terminated before complete ROCm purge/reinstall lifecycle;
- `yay`/sudo ticket timeout issues in long AUR workflows;
- missing vLLM dependency cascades and engine-core failures;
- DeepSpeed benchmark null/empty result scenarios;
- integrated GPUs leaking into visible device environment variables;
- benchmark export behavior lacking enough end-user feedback/context.

## **Scope**

### **1) ROCm Installer and Force-Reinstall Lifecycle**

- `scripts/install_rocm.sh`
  - forced purge engines per package family (`apt/dpkg`, `dnf/yum`, `zypper`, `pacman`);
  - force-reinstall state machine with reboot choreography;
  - resume-after-reboot launcher/autostart handling;
  - mandatory second reboot finalization.

### **2) Arch/CachyOS AUR Flow Hardening**

- `scripts/install_rocm.sh`
  - AUR helper executes as non-root user;
  - repo packages install through `pacman`;
  - AUR package availability checks before install;
  - sudo ticket prime + keepalive for extended helper runtime.

### **3) Runtime Environment, Multi-GPU, and iGPU Filtering**

- `scripts/setup_permanent_rocm_env.sh`
- `scripts/lib/benchmark_common.sh` (new shared benchmark runtime layer)
- `rusty-stack/src/benchmarks/mod.rs`
- `rusty-stack/src/widgets/benchmarks_page.rs`
  - strengthened iGPU detection heuristics;
  - discrete-only visible device export for mixed systems;
  - fish/bash/zsh-compatible environment integration;
  - managed writable Triton cache paths.

### **4) vLLM/DeepSpeed/Megatron Reliability**

- `scripts/install_vllm_multi.sh`
- `scripts/run_vllm_benchmarks.sh`
- `scripts/run_deepspeed_benchmarks.sh`
- `scripts/install_megatron.sh`
- `scripts/run_megatron_benchmarks.sh` (new)
- `scripts/run_all_benchmarks_suite.sh`
  - runtime preflight and missing-dependency reconciliation;
  - safer benchmark execution and output capture;
  - Megatron benchmark integration into suite flow.

### **5) Rusty-Stack Benchmark Parsing and UX**

- `rusty-stack/src/benchmark_logs.rs` (new)
- `rusty-stack/src/installer.rs`
- `rusty-stack/src/app.rs`
- `rusty-stack/src/widgets/benchmarks_page.rs`
  - robust JSON extraction from mixed benchmark logs;
  - benchmark screen export feedback;
  - richer HTML report generation (charts + tables + details).

## **Benchmark `E` Export (User-Facing Detail)**

On the benchmark screen, pressing `E` now:

- exports a full HTML benchmark report;
- emits **success/failure notification** in the TUI;
- includes the output path for immediate user retrieval;
- renders detailed visuals:
  - animated line charts,
  - axis labels/ticks,
  - plotted data points,
  - summary/metrics/samples/GPU tables.

Default report output: `~/.mlstack/reports/benchmark_report_<timestamp>.html`.

## **Documentation Updated**

- `README.md` (benchmark export workflow and report details)
- `CHANGELOG.md` (Unreleased release-track status + stabilization details)
- `docs/core/rocm_installation_guide.md`
- `docs/extensions/vllm_guide.md`
- `docs/rusty_stack_guide.md`
- `docs/guides/troubleshooting_guide.md`
- `docs/INSTALLER_STATUS.md`
- `docs/PR_ROCM_STABILIZATION_2026-02.md` (this PR specification)

## **Validation Evidence**

Executed:

- `cargo check --manifest-path rusty-stack/Cargo.toml -q`
- `bash -n scripts/install_rocm.sh scripts/setup_permanent_rocm_env.sh scripts/lib/benchmark_common.sh scripts/install_vllm_multi.sh scripts/run_vllm_benchmarks.sh scripts/run_deepspeed_benchmarks.sh scripts/run_megatron_benchmarks.sh`

Observed runtime evidence:

- vLLM benchmark JSON success with throughput and discrete-only visible devices (`visible_devices=0,1`) in latest validated run logs.

## **Risk Assessment**

- **Medium**: ROCm reinstall flow is system-level and broad in impact.
- **Low-to-medium**: iGPU filtering heuristics may require edge-case tuning on unusual mixed-GPU topologies.
- **Low**: benchmark parser/export UX changes are additive.

## **Rollback Plan**

1. Revert PR commit range.
2. Re-run prior installer paths without force reinstall.
3. Regenerate `~/.mlstack_env`.
4. Re-run baseline verification/benchmark commands.

## **Merge Checklist**

- [ ] CI/build checks pass.
- [ ] Maintainer spot-tests force reinstall on at least one distro per package family.
- [ ] Maintainer validates benchmark `E` export notification and output file path behavior.
- [ ] Maintainer validates vLLM/DeepSpeed/Megatron benchmark output parsing.
- [ ] Maintainer confirms docs/changelog alignment with implemented behavior.
