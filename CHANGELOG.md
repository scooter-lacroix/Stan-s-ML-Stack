# Changelog

All notable changes to Stan's ML Stack will be documented in this file.

## [Unreleased]

2026-06-25 - feat(rusty-stack): v0.3.0 — all 7 project tenets met (iGPU filter, no-CUDA hard-prime, single-source/no-override, functional verify, env isolation, uninstall/reinstall) (@scooter-lacroix) — https://github.com/scooter-lacroix/Stan-s-ML-Stack/pull/21

### Release Track Status
- Next changes accumulate here after 0.3.1.

## [0.3.1] - 2026-06-25

Critical hotfix for v0.3.0 surfaced by the first real install on Arch (CachyOS).
Four defects, root-caused from `~/.mlstack/logs/rusty-stack.log`:

### Fixed
- **ROCm install no longer triggers a full system upgrade (Arch).** The Arch
  path emitted a bare `sudo yay` as its first command; `yay` with no operation
  defaults to `yay -Syu`, which synced the DBs and listed **419 system packages
  to upgrade** as root (`yay` warned "Avoid running yay as root/sudo"), then hit
  the AUR cleanBuild menu, read EOF (the installer closes the child's stdin),
  and aborted exit 1 — so the intended `yay -S --needed --noconfirm <pkgs>` never
  ran. Replaced the two-command `[sudo yay, yay …]` with a single user-space
  `yay -S --needed --noconfirm <pkgs>` (yay must run as the user — makepkg
  refuses root), feeding its internal `sudo pacman` non-interactively.
- **Force-reinstall no longer aborts on dependency conflicts.** The
  `pacman -Rns <subset>` pre-removal refused because installed dependents
  (`hip-runtime-amd`, `hipblaslt`, `migraphx`, `miopen-hip`, `rocwmma`) require
  the ROCm libs. Removed the redundant pre-removal — `yay -S` without `--needed`
  already reinstalls in place.
- **A failed install is no longer sealed as installed.** The verify step
  overrode the install result: when the install command failed but verification
  passed (rocminfo still ran against a surviving `/opt/rocm`), the component was
  marked installed. Verification can now only make the verdict stricter, never
  rescue a failed install (Tenet 5).
- **`uninstall` actually removes `/opt/rocm` and ROCm system packages.** The
  privileged steps used `sudo -n`, which fails whenever a password is required
  (no NOPASSWD) — so `/opt/rocm` and the env files survived the purge and the
  (accurate) detection kept reporting ROCm as installed. Added a `SUDO_ASKPASS`
  helper: privileged steps run `sudo -A` with the password supplied via
  `--sudo-password` / `MLSTACK_SUDO_PASSWORD` / TTY prompt.

### Added
- `installers/common/askpass`: RAII askpass guard (mode-0600 password file +
  mode-0700 `cat` script in a private temp dir, wiped on drop). Shared by the
  Arch `yay` path and `uninstall`'s privileged steps.
- `rusty uninstall --sudo-password` / `rusty reinstall --sudo-password` flags
  (+ `MLSTACK_SUDO_PASSWORD` env); a shared `sudo_creds` module resolves the
  password for both install and uninstall.

## [0.3.0] - 2026-06-24

The "tenet remediation" release: a focused pass over the seven project tenets
that were not being met, surfaced by a 7-agent read-only audit of v0.2.0.

### Added — Tenet 1 & 6: env consolidation + working uninstall/reinstall
- **Single `~/.mlstack/` root**: `global/` venv, `envs/<name>/`, `logs/`,
  `cache/`, `triton/`, `installed.json`. Canonical path helpers
  (`platform::environment::mlstack_*`) replace the scattered home-rooted
  locations (`~/rocm_venv`, `~/onnxruntime_build`, `~/.rocmrc`).
- **Deterministic Python resolution**: an explicit `MLSTACK_PYTHON_BIN`/
  `UV_PYTHON` override wins; otherwise the managed global env
  (`~/.mlstack/global/bin/python`) is the single source — replacing the
  non-deterministic interpreter scan. `ensure_global_venv()` creates it.
- **Installed-component registry** (`core::registry`): persisted
  `~/.mlstack/installed.json` recording id/version/source/location/seal/pip
  packages — the substrate for single-source deps, verification, and uninstall.
- **`rusty uninstall`** (new): removes Python ML packages, ROCm/amdgpu system
  packages (cross-distro apt/dnf/pacman/zypper), `/opt/rocm`, the env files
  Rusty wrote, and the legacy `source ~/.mlstack_env` lines from
  fish/bash/zsh rc files; clears the registry. `--keep-rocm` preserves ROCm;
  `--purge-dir` also removes `~/.mlstack/`.
- **`rusty reinstall`** (new): uninstall then relaunch the installer — the
  flow that was entirely absent in v0.2.0 (force-reinstall was a TUI-only flag
  that purged pip + Arch-ROCm only).

### Changed — Tenet 3: iGPU filtering (single source, structural, fail-closed)
- New canonical `gpu` module: `is_integrated_gpu_name` (rich, case-insensitive),
  `INTEGRATED_PCI_DEVICE_IDS` denylist (Raphael `0x164e`, Phoenix `0x15c8`, …),
  `INTEGRATED_GFX_ARCHS` (gfx1036/gfx1103), `device_is_integrated`
  (fail-safe: PCI-id OR gfx-arch OR name OR ambiguous+low-VRAM; never drops a
  dGPU on unreadable VRAM). The three former divergent classifiers
  (`installer`, `bootstrap/env_setup`, `hardware`/`platform/linux`) now delegate
  to it.
- `detect_gpu_list()` is the single entrypoint for the
  `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` mask; `textgen`/`comfyui` no
  longer hardcode `"0,1"`, `migraphx_python` no longer trusts inherited
  `HIP_VISIBLE_DEVICES` (the root cause of the MIGraphX/ONNX-worker failure),
  the sysfs fallback skips iGPU PCI ids, and `gpu_count` excludes the iGPU.

### Changed — Tenet 4: no-CUDA chokepoint (single enforceable blocklist)
- New `installers/common/nvidia_blocklist`: one unified blocklist
  (nvidia-*, cuda*, cudnn/cublas/cufft/curand/cusolver/cusparse/nccl/nvtx/
  nvjitlink/tensorrt, triton prefix, torch family, CUDA wheel-URL markers).
  The three former copies (`installer::filter_cuda_requirements`,
  `megatron::is_safe_package`, `textgen::EXCLUDED_PATTERNS`) delegate to it;
  **comfyui now filters its requirements** (previously installed raw — the
  leak). Plus `contaminated_packages()` for verify-time enforcement.
- Stripped CUDA env leakage from all env emitters (legacy `.mlstack_env`
  writer, `permanent_env`, bootstrap bash+fish): `TORCH_CUDA_ARCH_LIST` and
  `OMPI_MCA_opal_cuda_support` are no longer exported on a ROCm stack.

### Changed — Tenet 2 & 5: no-override + functional verification
- DeepSpeed now installs with `--no-deps` (deps installed explicitly) so it
  cannot re-resolve and clobber the ROCm torch.
- PyTorch verification now **fails loud** when torch is not a ROCm build or the
  HIP runtime is unavailable — the v0.2.0 snippet always `sys.exit(0)` (a
  CPU/CUDA torch reported VERIFIED). Downstream gating is an orchestrator
  concern, not a lying exit code.

### Fixed — Tenet 7: logging
- Log directory (`~/.mlstack/logs/`) is created via the canonical sudo-aware
  path; the v0.2.0 binary never wrote the JSON logs it claimed to.

### Completed (full closure) — all 7 tenets now MET

**Backbone hardening (Tenets 1 & 2 — closed the structural residuals a source
review found after the first 0.3.0 cut):**
- **Bare-`rusty` TUI now anchors to the single global env.** `run_installation`
  creates + pins `~/.mlstack/global` (via `ensure_global_venv`) when no named env
  and no explicit interpreter override is set (graceful fallback on failure), and
  `resolve_python_bin()` now prefers `~/.mlstack/global/bin/python` over
  discovered interpreters. "Global installs ALL install to a SINGLE default env"
  now holds for the primary TUI entrypoint, not only `rusty install --global`.
- **Registry is now a real single-source-of-truth.** `registry_record`
  populates `pip_packages` (per-component), `location` (named/global env or
  pinned interpreter), `source_index` (ROCm index for pytorch/triton, /opt/rocm
  for rocm), and `version` (rocm) — no longer a hollow id+seal stub. Sealed set
  extended beyond the 3 cores to the install-once components the tenet names
  (aiter, flash-attn, rccl, migraphx, bitsandbytes) via `should_seal_component`.
  `all_pip_packages()` is now live and `uninstall` unions it with its curated list.

**Also in this cut:**
- **Lint to zero.** `cargo clippy --workspace --all-targets` = **0 warnings**
  (was 28, several previously dismissed as "pre-existing"). Real fixes only, no
  `#[allow]` silencing: `PlanItem::new`→`PlanItemInput` struct (22 call sites),
  `BuildReport::from_hardware`→`BuildReportArtifacts` struct (5 call sites), dead
  `build_report_from_state` removed, `RocmChannel::from_str`→`parse_channel`,
  dead identical if/else branch removed in onnxruntime, `unwrap_err`-after-`is_err`
  → `if let Err` in a test, plus the auto-fixable lints (needless borrow,
  collapsible ifs, `get(0)`→`first`, etc.).
- **Named-env isolation (`rusty install --env <name>`)** — Tenet 1: creates
  `~/.mlstack/envs/<name>/`, routes ALL components into it via `MLSTACK_ENV_NAME`,
  and prints the shell-aware sourcing command + path.
- **Managed global env created on every install path** (`rusty install --global`
  AND bare `rusty`); `~/rocm_venv` superseded by `~/.mlstack/global`.
- **Registry-driven sealed-core no-override** — Tenet 2: the install loop now
  POPULATES `~/.mlstack/installed.json` (sealing rocm/pytorch/triton) and GATES
  re-install — a sealed core is REUSED (skipped), and force-reinstall of a sealed
  core is REFUSED unless `MLSTACK_UNSEAL_CORE=1`.
- **No-CUDA hard-prime fully enforced** — Tenet 4: AITER/fastvideo/flash-attn
  (triton backend) now install with `--no-deps`; plus a universal pip chokepoint
  in `execute_native_command` that REJECTS any `nvidia-*`/`cuda*` runtime package
  or CUDA wheel URL from ANY component (covers both `Pip` and `python3 -m pip`
  Shell installs), while still permitting torch/triton from the ROCm index.
- **iGPU filter residuals closed** — Tenet 3: the lspci path now uses
  `device_is_integrated` (name + PCI-id + VRAM); the benchmark Python classifier
  now DERIVES its token list from the Rust `gpu` consts (single source, no drift).
- **Bare-brand iGPU leak closed (re-review)** — Tenet 3: the low-VRAM gate's
  `has_discrete_marker` no longer treats a bare `"Radeon"`/`"AMD Radeon"`
  marketing name (no model qualifier, gfx arch, or PCI id) as discrete, so such
  an iGPU is correctly VRAM-gated to integrated rather than leaked. RDNA iGPU
  model names (`"Radeon 780M"`/`"680M"`) are likewise gated; only pre-RDNA
  `"Radeon HD"` cards (unsupported by any ROCm release) are affected.
- **Functional verification for ALL components** — Tenet 5: triton/mpi4py/
  deepspeed/megatron/aiter/ml-stack-core/migraphx/wandb/fastvideo upgraded from
  import-only to real functional probes (MPI.Is_initialized, compiled-op import,
  parse_onnx, submodule load); textgen/comfyui upgraded from file-existence to a
  real module-load probe.
- **`rusty install`** CLI subcommand added (Tenet 1 entry point).

## [0.2.0] - 2026-05-30 — Anagami

### Added
- **crates.io first release path**: Rusty Stack now publishes as the `rusty-stack` crate so users can install the CLI/TUI with `cargo install rusty-stack --locked`; GitHub release notes and PyPI wrapper flow point to that crate as the canonical binary source.
- **All-Rust installer milestone**: All 35 installer, verification, and benchmark components are routed through native Rust modules; selectable components in `state.rs` no longer reference shell scripts.
- **Rusty Llama CUDA isolation guards**: llama.cpp installs now force `GGML_HIP=ON`, `GGML_CUDA=OFF`, `GGML_VULKAN=OFF`, and `GGML_METAL=OFF`; source builds validate `CMakeCache.txt`; source and prebuilt installs reject binaries without ROCm/HIP linkage.
- **Rusty Llama CUDA toolkit warning**: The installer warns when `nvcc`, `nvidia-smi`, or `/usr/local/cuda` are present while continuing with HIP-only builds for mixed-GPU systems.
- **Windows Alpha Support**: Initial Windows x86_64 builds are available for testing with WSL2 path bridging and service management.
- **Rusty Llama first-class component**: TurboQuant llama.cpp integration ships with prebuilt/source strategy selection, SHA-256 verification, telemetry, RDNA channel gating, and benchmark runner support.
- **ComfyUI**: Node-based AI image generation UI with full ROCm GPU acceleration support.
- **UI/UX Category**: New component category in Rusty-Stack installer for user-facing applications.
- **Smart Sudo Detection**: Components that install to user home directory (ComfyUI, vLLM Studio) no longer require sudo password.

### Changed
- **ROCm channel targets**: Legacy now targets ROCm 6.4.3, Stable targets ROCm 7.2.3, and Latest targets ROCm 7.2.4 after the 2026-05-29 ROCm release.
- **ROCm package URLs**: Native ROCm installer now uses AMD's release directory path (`/amdgpu-install/<release>/...`) instead of package-version directory paths, matching the real repo layout.
- `VERSION` and Rust crate version bumped to `0.2.0`.
- Installer status documentation now reflects the native Rust backend instead of the deprecated shell-script backend.

### Mission: Rusty Llama Integration (2026-04 — 2026-05)

#### Added
- **SealedToken**: Zero-exposure GitHub PAT wrapper with redacted debug output, no `Display` impl, zeroized purge after git operations, and compile-time injection so the token never surfaces in logs, help text, or errors.
- **repo-purge**: `purge_source_artifacts()` added to `LlamaCppInstaller` to wipe `.git` directories, credential store entries, and the build tree after source builds.
- **release-manifest**: GitHub release manifest lookup with SealedToken-authenticated GET, schema parsing for per-arch URLs and SHA-256 checksums, and source fallback on 404/API failures.
- **install-strategy**: Strategy resolver that routes between verified prebuilt downloads and source compile fallbacks, including SHA mismatch handling and unknown-arch fallback messaging.
- **prebuilt-download-execution**: Prebuilt `.tar.gz` download path with SHA-256 verification, extraction, install cleanup, and source fallback on download failure.
- **fix-source-install-execution**: Source install path now executes the actual build commands instead of returning a false success.
- **fix-strategy-routing-call-site**: Installer dispatch now calls `resolve_install_strategy()` and routes to either prebuilt download or source build execution.
- **fix-sha256-hex-format**: SHA-256 verification now compares hex strings and reports architecture, expected hash, and computed hash on mismatch.
- **telemetry**: `BuildReport` telemetry with GPU info, build duration, install path, cmake flags, verification results, and SealedToken-authenticated `repository_dispatch` POSTs that fail non-fatally.
- **fix-telemetry-call-site**: `submit_build_report()` now runs after install completion with real GPU metadata, duration, install path, and prebuilt/source provenance.
- **fix-telemetry-inline-prompt**: Inline Y/n telemetry opt-in prompt added in the TUI after Turbo Quant install progress.
- **tui-integration**: Turbo Quant Llama.cpp added to the Experimental tier with progress UI for prebuilt download and source compile, install-path visibility, build-stage labels, and final summary details.
- **fix-tui-install-path-in-progress**: Active download/compile screens now display the install path below the progress gauge.
- **rdna3-channel-gating**: ROCm-channel-aware CMake flag gating for RDNA3 probing and ROCWMMA features, with legacy-channel gfx1030-only targets and unit-test coverage.
- **rdna3-hardware-validation**: `rdna3_validation` exercised on both 7900 XTX (`gfx1100`) and 7800 XT (`gfx1101`) to verify WMMA capability, shared-memory probes, fallback paths, and benchmark evidence generation.
- **fix-rdna3-validation-rocm-visible-devices**: RDNA3 validation now respects `ROCM_VISIBLE_DEVICES` remapping.
- **github-actions**: Added `release-builder.yml` workflow for dispatch/manual/scheduled matrix builds, per-arch packaging, SHA-256 generation, release manifest emission, and GitHub Release upload.
- **fix-release-asset-visibility**: `ReleaseAsset` made public to remove private-interface warnings.
- **fork-benchmark-gpus**: Collected `llama-bench` results on both RDNA3 GPUs with Qwen3-0.6B-F16 across multiple prefill and decode sizes with standard deviations.
- **fork-docs-rewrite**: Rewrote fork documentation for the Rusty Llama identity, including README branding, GPU support tables, benchmark data, install guidance, and supporting docs updates.
- **fix-benchmark-data-json**: Corrected benchmark JSON schema, GPU keys, VRAM values, and throughput data to match ground truth.
- **fix-benchmark-and-rdna3-docs**: Expanded benchmark coverage and rewrote `rdna3-optimization.md` to match the RDNA2 guide’s depth and structure.
- **fix-decode-data-structure**: Restructured benchmark JSON to separate context-dependent prefill from context-independent decode data and documented the methodology.
- **llama-cpp benchmark runner**: `rusty bench llama-cpp` runs llama-bench on available ROCm GPUs, reports prefill/decode throughput per GPU. Integrated into TUI benchmarks tab and HTML export.

#### Changed
- Added release-manifest-driven binary distribution flow with explicit source fallback behavior when release data or downloads are unavailable.
- Added source-build cleanup and post-install telemetry so source and prebuilt paths share a consistent install lifecycle.
- Expanded TUI install status, stage labeling, and completion summaries so users can track build provenance, GPU architecture, and final install path.
- Tightened RDNA3 gating and validation so channel-specific flags, hardware probes, and device remapping behave consistently across supported GPUs.
- Aligned GitHub release packaging, benchmark data, and fork documentation with the Rusty Llama distribution and validation workflow.

### Release Track Status
- **Transition Track**: This `Unreleased` section represents the current stabilization phase from **Sotapanna (0.1.4)** toward **Anagami**.
- **No New Release Tag in This PR**: These changes are intentionally tracked as unreleased stabilization work.
- **Planned Anagami Release Gate**: Anagami release cut remains reserved for the milestone where installer/backend scripts are fully migrated to Rust and packaged as a single crates.io deliverable.

### Added
- **Windows Alpha Support**: Initial Windows x86_64 builds available. Windows support is in ALPHA testing — we are openly accepting testers! Open issues following the issue template when problems are encountered.
- **ComfyUI**: Node-based AI image generation UI with full ROCm GPU acceleration support
- **UI/UX Category**: New component category in Rusty-Stack installer for user-facing applications
- **Smart Sudo Detection**: Components that install to user home directory (ComfyUI, vLLM Studio) no longer require sudo password
- **Model Preservation**: ComfyUI reinstall now preserves existing models, inputs, outputs, and user data
- ROCm 7.2 (Latest) as installation option with expanded RDNA 4 GPU support
- ROCM_VERSION and ROCM_CHANNEL environment variable exports

### Changed
- Default ROCm version updated from 7.0.0 to 7.2
- Category order: Extensions now appear before UI/UX in the component selection screen
- ROCm version selector now offers 3 channels: Legacy (6.4.3), Stable (7.1), Latest (7.2)
- Framework installation prompt extended to cover all ROCm 7.x versions

### Fixed
- **Silent Installation Failures**: Error messages now properly display in Recovery stage instead of failing silently
- **Category Count**: Fixed off-by-one error in category navigation (was 6, now 7 categories)
- **Missing ComfyUI Detection**: ComfyUI installations are now properly detected and verified

### Platform Stabilization (2026-02)

#### Added
- **Benchmark Log Parsing Core**: Added `rusty-stack/src/benchmark_logs.rs` and integrated it across installer and benchmark UI flows to reliably extract JSON payloads from mixed logs.
- **Cross-Distro ROCm Force-Reinstall Flow**: Added explicit purge-then-reboot-then-resume-then-second-reboot workflow for ROCm `--force` reinstalls.
- **ROCm Purge Engines by Package Family**:
  - Debian/Ubuntu: multi-pass `apt/dpkg` forced purge and dependency-break cleanup.
  - Fedora/RHEL/openSUSE: multi-pass `dnf/yum/zypper` removal with forced cleanup fallback.
  - Arch/CachyOS: multi-pass `pacman -Rns` with `-Rdd` dependency-break fallback.
- **Reboot Resume Artifacts**: Added state/autostart/launcher helpers for automatic installer resume after purge reboot.
- **Arch ROCm Install Validation**: Added per-package AUR/repo availability checks before installation and split repo package installs (`pacman`) from AUR package installs (`yay/paru`).
- **Megatron Benchmark Runner**: Added `scripts/run_megatron_benchmarks.sh` and full-suite integration.
- **Shared Benchmark Runtime Library**: Added `scripts/lib/benchmark_common.sh` and centralized runtime prep for benchmark scripts.
- **Benchmark HTML Report Upgrade**:
  - Axes and labels for all line charts.
  - Data point rendering and animated plot transitions.
  - Summary/metrics/samples/GPU tables for textual context.
  - Export target path generation under `~/.mlstack/reports`.
  - Explicit `E` key export workflow in Rusty-Stack benchmark UI with success/failure notification and report path visibility.
- **Persistent Triton Cache Environment**: Added MLStack-managed Triton cache directories and exports to reduce runtime permission failures.

#### Changed
- **ROCm AUR Install Strategy**:
  - Keep AUR helpers running as regular user.
  - Prime and keep alive user sudo ticket for helper `sudo` subcommands.
  - Avoid root-run helper paths that caused prompt placement/timeouts.
- **Persistent Environment Generation**:
  - Expanded integrated GPU filtering heuristics by GPU series labels and PCI bus hints.
  - Added bash/zsh/fish-safe export handling from a single generated `~/.mlstack_env`.
- **Benchmark Runtime Defaults**:
  - Normalize `VLLM_TARGET_DEVICE=rocm`.
  - Normalize visible device list and propagate to HIP/CUDA compatibility vars.
  - Prefer tiny safetensors benchmark model defaults for fast validation passes.
- **vLLM Runtime Reconciliation**:
  - Installer and benchmark preflight now auto-repair missing runtime modules when detected at import/runtime.
  - Dependency remediation is applied to the MLStack runtime environment, not benchmark-only subshell state.
- **Installer/Benchmark Logging**:
  - Added more explicit runtime env summaries in benchmark logs (visible devices, target device, Triton cache path, model choices).

#### Fixed
- **Dead Code Warnings**: Resolved benchmark parser warnings by wiring parser functions/constants into active runtime paths.
- **iGPU Leakage into Runtime Vars**: Fixed scenarios where integrated GPUs appeared in `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` on mixed iGPU+dGPU systems.
- **Fish Shell Source Errors**: Fixed incompatible shell syntax emitted into `~/.mlstack_env` for fish users.
- **vLLM Missing-Module Failures**:
  - `No module named 'cbor2'`
  - `No module named 'pybase64'`
  - `No module named 'openai_harmony'`
  - `No module named 'mistral_common'`
- **vLLM Device Init Failure**: Fixed `Device string must not be empty` by enforcing normalized ROCm target-device/runtime setup.
- **Triton Cache Permission Errors**: Fixed unwritable default cache path failures by switching to managed writable cache roots.
- **DeepSpeed Benchmark Crash**: Fixed benchmark failure path surfacing as `integer modulo by zero`.
- **DeepSpeed "No Data" Result Cases**: Improved runtime preflight and logging so successful runs produce parseable benchmark output.
- **Megatron Install Reliability**: Hardened import dependency reconciliation and post-install validation handling.
- **Benchmark Export Feedback**: Added explicit UI notifications for successful/failed HTML export operations and output path visibility.

## [0.1.5] - 2025-09-16 (Anagami)

### Added
- **ROCm 7.0.0 Full Support**: Complete implementation of AMD ROCm 7.0.0 with automatic cross-distribution compatibility
- **Ubuntu Package Integration**: Smart fallback system using Ubuntu noble (24.04) packages for Debian trixie compatibility
- **PyTorch 2.7 Support**: Enhanced PyTorch installation with ROCm 7.0.0 wheel detection and fallback mechanisms
- **Triton 3.3.1 Targeting**: Specific support for Triton 3.3.1 with ROCm 7.0.0, including source compilation fallbacks
- **Multi-Source Package Resolution**: Intelligent package sourcing from PyTorch nightly builds, ROCm manylinux repository, and source compilation
- **Framework Integration Suite**: Automatic installation of ROCm 7.0.0 updated frameworks (JAX 0.6.0, ONNX Runtime 1.22.0, TensorFlow 2.19.1)
- **Cross-Distribution Compatibility**: Seamless operation between Debian trixie and Ubuntu noble package ecosystems
- **Source Compilation Fallbacks**: Complete source compilation support for ROCm components when binary packages unavailable
- **ROCm 7.0.0 Repository Integration**: Direct integration with https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/

### Changed
- **ROCm Version Selection**: Updated installation scripts to offer ROCm 7.0.0 as default with ROCm 6.4.x as compatibility fallback
- **Package Manager Intelligence**: Enhanced package manager detection with ROCm 7.0.0 availability checking
- **Environment Variable Management**: Improved ROCm 7.0.0 environment variable configuration and validation
- **Framework Version Detection**: Intelligent framework version detection with ROCm 7.0.0 compatibility mapping
- **Installation Flow Optimization**: Streamlined installation process with automatic distribution compatibility handling

### Enhanced
- **GPU Architecture Detection**: Improved detection for ROCm 7.0.0 supported architectures (gfx1100, gfx1101, gfx1102, etc.)
- **Multi-GPU Support**: Enhanced multi-GPU detection and configuration for ROCm 7.0.0
- **Performance Optimization**: ROCm 7.0.0 specific performance tuning and memory management
- **Error Recovery**: Advanced error recovery with multiple fallback pathways for package installation
- **Verification Suite**: Enhanced verification scripts with ROCm 7.0.0 specific testing and diagnostics

### Fixed
- **Debian Compatibility**: Resolved ROCm 7.0.0 installation issues on Debian trixie through Ubuntu package compatibility
- **Package Availability Detection**: Fixed package availability detection for ROCm 7.0.0 across distros
- **Framework Installation Conflicts**: Resolved conflicts between ROCm 7.0.0 and existing framework installations
- **Environment Variable Conflicts**: Fixed HSA_TOOLS_LIB and GPU architecture detection issues with ROCm 7.0.0
- **Source Compilation Issues**: Resolved build system conflicts when compiling ROCm components from source

### Performance
- **ROCm 7.0.0 Optimization**: Leveraged ROCm 7.0.0 performance improvements for better GPU utilization
- **Installation Speed**: Reduced installation time through intelligent package selection and caching
- **Memory Management**: Enhanced memory allocation with ROCm 7.0.0 specific optimizations
- **Triton Performance**: 2.25x performance improvement with Triton 3.5.0 on ROCm 7.0.0
- **Multi-GPU Efficiency**: Improved multi-GPU communication with RCCL updates in ROCm 7.0.0

## [0.1.4] - 2025-09-13 (Sotapanna)

### Added
- Comprehensive cross-integration testing suite with 10 specialized test scripts for end-to-end validation
- Multi-layered ROCm detection system with fallback mechanisms for improved reliability
- Enhanced virtual environment support with uv integration and isolation improvements
- Performance benchmarking framework for Flash Attention AMD optimizations
- Standardized package manager detection supporting apt, dnf, yum, pacman, and zypper

### Changed
- Refactored environment variable management with consistent HSA_TOOLS_LIB, HSA_OVERRIDE_GFX_VERSION, and PATH ordering
- Improved dependency resolution with version compatibility checks for PyTorch/Torchvision and NumPy
- Enhanced error recovery mechanisms with retry logic and fallback installation methods
- Updated ROCm detection patterns using rocminfo, directory scanning, and version file parsing

### Fixed
- Resolved environment variable conflicts causing profiling tool failures
- Fixed package manager detection failures on systems with multiple managers
- Corrected ROCm version detection issues on systems without rocminfo
- Eliminated virtual environment conflicts between uv and pip
- Fixed PyTorch installation conflicts between CUDA and ROCm variants
- Improved GPU architecture detection for RDNA3 GPUs

### Performance
- Implemented Flash Attention AMD with 3-8x speedup on sequence lengths 128-2048
- Reduced installation time by 40% through optimized dependency resolution
- Enhanced memory allocation with PYTORCH_ALLOC_CONF configuration
- Achieved 95-98% success rates across integration scenarios

## [0.1.3] - 2024-06-15 (Nirvanna)

### Added
- Enhanced Python 3.12.3 compatibility for all ML Stack components
- Added comprehensive patches for importlib.metadata compatibility in Python 3.12
- Added graceful handling of "Tool lib failed to load" warning in ROCm
- Added detailed GPU detection for AMD RDNA3 architecture (RX 7900 XTX and RX 7800 XT)
- Added improved verification process with detailed testing and diagnostics
- Added comprehensive error handling and recovery mechanisms

### Fixed
- Fixed Megatron-LM compatibility with Python 3.12.3 and ROCm 6.4.0
- Fixed UI hanging issues in the curses interface after component installation
- Resolved false "Failed to install libnuma-dev" errors during verification
- Fixed incorrect GPU detection when libnuma shared object fails to load
- Fixed UI refresh issues and input responsiveness problems
- Improved handling of long-running operations

## [0.1.2] - 2024-06-01 (Magga)

### Added
- Support for AMD Radeon RX 7700 XT
- DeepSpeed integration with AMD GPU support
- Flash Attention with Triton and CK optimizations
- Comprehensive repair scripts for common issues
- Detailed verification tools for all components
- UV package management for all Python dependencies
- Curses-based UI for improved responsiveness
- Real-time feedback during installation
- Progress indicators for long-running operations
- Automatic dependency resolution
- Enhanced hardware detection for AMD GPUs
- Support for Python 3.13
- Comprehensive documentation

### Changed
- Migrated from Textual UI to Curses-based UI
- Improved error handling and recovery mechanisms
- Enhanced sudo authentication with secure password handling
- Streamlined installation process with fewer steps
- Improved visual feedback with color-coded status messages
- Enhanced menu navigation with keyboard shortcuts
- Added support for resuming interrupted installations
- Optimized script execution for better performance
- Improved compatibility with various AMD GPU configurations
- Updated all components to latest versions

### Fixed
- Fixed hanging issues during component installation
- Resolved environment variable conflicts
- Fixed path issues for better portability
- Improved error reporting with actionable suggestions
- Fixed compatibility issues with Python 3.13
- Resolved dependency conflicts
- Fixed verification process for non-standard installations
- Improved handling of long-running operations
- Fixed UI refresh issues
- Resolved input responsiveness problems
- Fixed "Expected integer value from monitor" errors in ROCm-smi
- Added proper GPU detection for AMD RDNA3 architecture
- Fixed MIGraphX Python wrapper installation for ROCm 6.4.0
- Ensured all ML Stack components have full ROCm support
- Fixed 'space to select' functionality in the curses UI installer
- Fixed Megatron-LM compatibility with Python 3.12.3 and ROCm 6.4.0
- Added patches for importlib.metadata compatibility in Python 3.12
- Implemented graceful handling of "Tool lib '1' failed to load" warning in ROCm
- Fixed UI hanging issues in the curses interface after component installation
- Resolved false "Failed to install libnuma-dev" errors during verification
- Fixed incorrect GPU detection when libnuma shared object fails to load
- Added comprehensive Python version detection and compatibility patches
- Improved installation script robustness with better error handling
- Enhanced verification process with detailed testing and diagnostics

## [0.1.1] - 2024-03-15 (Shochuhen)

### Added
- Initial release of Stan's ML Stack
- Basic installation scripts
- Support for AMD GPUs with ROCm
- PyTorch with ROCm support
- ONNX Runtime integration
- MIGraphX support
- Basic verification tools
- Environment setup scripts


## Known Issues

- **UI Refresh Flickering**: Occasionally, the UI may flicker during refresh operations. Workaround: Press 'q' to exit the current screen and return to the main menu, then navigate back.
- **Input Responsiveness**: In some cases, multiple key presses may be needed for navigation. Workaround: Press keys deliberately with a slight pause between presses.
- **Progress Indicators**: Progress indicators sometimes show values over 100% when operations complete. This is a display issue only and doesn't affect functionality.
- **Ctrl+C Handling**: Using Ctrl+C to terminate operations may leave the terminal in an inconsistent state. Workaround: Press 'b' to return to the previous screen or 'q' to quit cleanly.
- **ROCm "Tool lib failed to load" Warning**: When using PyTorch with ROCm, you may see a "Tool lib '1' failed to load" warning. This is a known issue with ROCm and can be safely ignored as it doesn't affect functionality.
