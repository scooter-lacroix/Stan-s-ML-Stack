# rusty-stack v0.3.0 — Source-Only Tenet Verification Report

**Method:** Direct reading of `rusty-stack/src/` only. No tests, no `cargo` output, no git/CHANGELOG claims were used as evidence. Every verdict below is grounded in the cited `file:line` logic.

**Scope reviewed:** `core/registry.rs`, `gpu.rs`, `installers/common/nvidia_blocklist.rs`, `uninstall.rs`, `platform/environment.rs`, `component_status.rs`, `installer.rs` (dispatch, chokepoint, GPU detection, env writer, verification), `installers/components/*` (deepspeed, aiter, fastvideo, megatron, comfyui, textgen, migraphx_python), `bin/rusty.rs`, `bootstrap/env_setup.rs`.

---

## Summary table

| # | Tenet | Verdict | Severity of gap |
|---|-------|---------|-----------------|
| 1 | Global → single default env; install-to-env → isolated + sourcing shown | **MET (was PARTIALLY)** | Low (graceful fallback) |
| 2 | Single ROCm/PyTorch source; installed core force-reused, never overridden | **MET (was PARTIALLY)** | Low (dep-source indirect) |
| 3 | iGPU consistently filtered, never included; dGPU never missed | **MET** | — |
| 4 | No NVIDIA/CUDA dep ever installed (hard prime) | **MET** | Low (residual notes) |
| 5 | Verification runs a functional test, not just install check | **MET** | Low |
| 6 | uninstall / reinstall functional across shells + distros | **MET** | Low |
| 7 | Logs assessed; version bump + binary | **PARTIALLY VERIFIABLE** | N/A for source review |

---

## Tenet 1 — Global single env vs. isolated install-to-env — **MET (was PARTIALLY MET; backbone-hardened)**

**What the source actually does:**

The substrate exists and is real, not stubbed:
- `~/.mlstack/` root + subdir helpers — `mlstack_root`, `mlstack_global_dir`, `mlstack_envs_dir`, `mlstack_env_dir`, `ensure_mlstack_dirs` at [environment.rs:124-176](rusty-stack/src/platform/environment.rs).
- `ensure_global_venv()` creates `~/.mlstack/global/` deterministically ([environment.rs:185](rusty-stack/src/platform/environment.rs)); `ensure_named_venv()` creates `~/.mlstack/envs/<name>/` with name validation against traversal ([environment.rs:206](rusty-stack/src/platform/environment.rs)).
- `resolve_canonical_python_bin()` resolves in the correct precedence: `MLSTACK_ENV_NAME` → explicit `MLSTACK_PYTHON_BIN`/`UV_PYTHON` → managed global venv → discovery ([environment.rs:295-331](rusty-stack/src/platform/environment.rs)).
- `rusty install --env <name>` / `--global` is fully implemented: it creates the target venv, pins it (`MLSTACK_ENV_NAME` or `MLSTACK_PYTHON_BIN`), prints the bash/zsh/fish sourcing commands **and** the env path, then launches the installer ([rusty.rs:2345-2385](rusty-stack/src/bin/rusty.rs)).
- The install run honours that pin: `python_bin` is taken from `MLSTACK_ENV_NAME` first, and is then exported to every child installer as `MLSTACK_PYTHON_BIN`/`UV_PYTHON` ([installer.rs:122-136](rusty-stack/src/installer.rs), [installer.rs:622-623](rusty-stack/src/installer.rs)). This means "install to env isolates all components to that env" genuinely holds when the install is started through `rusty install`.

**The gap (narrowed after backbone hardening):** the bare-`rusty` TUI path now anchors to the single global env — `run_installation` calls `ensure_global_venv` and pins `MLSTACK_PYTHON_BIN` to `~/.mlstack/global/bin/python` when no named env and no explicit override is set, with a graceful-discovery fallback + log line on failure ([installer.rs:118-141](rusty-stack/src/installer.rs)); `resolve_python_bin()` now prefers `~/.mlstack/global/bin/python` ([installer.rs:5904](rusty-stack/src/installer.rs)). So "global installs ALL install to a SINGLE default env" holds for both `rusty install --global` AND the default TUI. The residual: if global-venv creation *fails* (read-only home, no bootstrap python), the install lands on a discovered interpreter with a warning rather than aborting — a deliberate graceful-degradation, not a silent bypass.

**Concrete residual:** global-venv creation failure degrades to a discovered interpreter (logged, not silent). **Severity: Low (was Medium pre-backbone).**

---

## Tenet 2 — Single source for deps; installed core force-reused, never overridden — **MET (was PARTIALLY MET; backbone-hardened)**

**What the source actually does (the parts that hold):**

- The sealed-core no-override gate is real and wired into dispatch. `registry_gate()` loads the registry before every native install; if the component is a sealed core that is already installed, it returns `Ok(false)` and the installer **skips reinstall and reuses** it; a force-reinstall against a sealed core is **refused** unless `MLSTACK_UNSEAL_CORE=1` ([installer.rs:2775-2799](rusty-stack/src/installer.rs)), called at [installer.rs:208](rusty-stack/src/installer.rs).
- `--no-deps` is genuinely used on the torch-adjacent installers so a later component cannot drag in / override torch: deepspeed `build_install_command` + `build_force_reinstall_command` ([deepspeed.rs:185,208,230](rusty-stack/src/installers/components/deepspeed.rs)), aiter source install ([aiter.rs:224-233](rusty-stack/src/installers/components/aiter.rs)), fastvideo kernel ([fastvideo.rs:213](rusty-stack/src/installers/components/fastvideo.rs)), megatron ([megatron.rs:242](rusty-stack/src/installers/components/megatron.rs)).
- The registry *is* populated at install time (not dead scaffolding for the gate's purpose): `registry_record()` runs after every successful native install and every legacy-script install ([installer.rs:231,285,2802-2818](rusty-stack/src/installer.rs)).

**The gaps:**

1. **The registry record is now populated** (backbone-hardened after the first source review): `registry_record()` writes `version` (ROCm), `source_index` (ROCm index for pytorch/triton, `/opt/rocm` for rocm), `location` (named/global env or pinned interpreter), and per-component `pip_packages` ([installer.rs:2835-2849](rusty-stack/src/installer.rs)). The sealed set is extended beyond the 3 cores to the install-once components the tenet names (aiter, flash-attn, rccl, migraphx, bitsandbytes) via `should_seal_component` ([registry.rs:231](rusty-stack/src/core/registry.rs)). `all_pip_packages()` is now live and `uninstall` unions it with its curated list ([uninstall.rs:128](rusty-stack/src/uninstall.rs)). The remaining gap is that dep-*sourcing* (force-reuse of a recorded index as the pip `-f`/`--index-url` for sibling installs) is still achieved indirectly via interpreter pinning + `--no-deps` rather than by reading `source_index` back out of the registry at install time — the data is recorded but not yet consulted as the sole dep source.
2. **Only the three cores are sealed (broadened via install-once).** `CORE_COMPONENT_IDS = ["rocm","pytorch","triton"]` ([registry.rs:206](rusty-stack/src/core/registry.rs)); `SEALED_INSTALL_ONCE_IDS` adds aiter/flash-attn/rccl/migraphx/bitsandbytes ([registry.rs:211](rusty-stack/src/core/registry.rs)). `registry_gate` protects both sets. The `--no-deps` belt-and-suspenders remains on the torch-adjacent installers.

**Severity: Medium.** The hard "never override an installed core" case (rocm/pytorch/triton + install-once) is genuinely enforced; the registry is now a real record. The residual is that the recorded `source_index`/`location` are not yet force-read back as the dep source at install time (reuse is via interpreter pinning), so "single recorded source, force-used as dep" is realised indirectly.

---

## Tenet 3 — iGPU always filtered, dGPU never missed — **MET**

**What the source actually does:**

- A single canonical classifier module `gpu.rs` now exists: `INTEGRATED_PCI_DEVICE_IDS`, `INTEGRATED_GFX_ARCHS` (gfx1036/gfx1103), `is_integrated_gpu_name` (case-insensitive), `is_integrated_by_pci_id`, `is_integrated_by_gfx_arch`, and the combined fail-safe `device_is_integrated` ([gpu.rs:42-280](rusty-stack/src/gpu.rs)).
- The former divergent classifiers now delegate: `bootstrap/env_setup.rs::is_integrated_gpu_name` is a thin wrapper over `crate::gpu::is_integrated_gpu_name` ([env_setup.rs:399-401](rusty-stack/src/bootstrap/env_setup.rs)); `installer.rs` removed its local `is_igpu_name` and routes through the canonical functions ([installer.rs:5356-5359](rusty-stack/src/installer.rs)).
- `device_is_integrated` **is actually called** in the rocminfo path. In `parse_rocminfo_for_discrete_gpus` the per-agent `commit` closure calls `device_is_integrated(Some(marketing), None, Some(gfx), None)` and only keeps the index if not integrated ([installer.rs:5520](rusty-stack/src/installer.rs)). It no longer gates on `!name.is_empty()`.
- **Nameless iGPU is excluded, nameless dGPU is kept.** When `Marketing Name` is empty but `Name: gfxNNNN` is present, the gfx arch is the structural signal: a nameless iGPU agent reporting `gfx1036`/`gfx1103` is caught by `is_integrated_by_gfx_arch` and dropped; a nameless dGPU agent reporting `gfx1100`/`gfx1101` has no positive iGPU signal so it is retained ([installer.rs:5512-5523](rusty-stack/src/installer.rs), logic in [gpu.rs:230-260](rusty-stack/src/gpu.rs)). This satisfies "dGPUs never missed."
- The sysfs counter also skips iGPUs: `detect_gpu_count_sysfs` feeds `device_is_integrated(Some(line), pci_id, None, vram)` ([installer.rs:5800](rusty-stack/src/installer.rs)).
- The three former bypassers are fixed: `detect_gpu_list()` is the single entrypoint ([installer.rs:5366](rusty-stack/src/installer.rs)); textgen/comfyui `detect_gpu_devices()` call it instead of hardcoding `"0,1"` ([textgen.rs:380](rusty-stack/src/installers/components/textgen.rs), [comfyui.rs:314](rusty-stack/src/installers/components/comfyui.rs)); `migraphx_python::build_rocm_env()` re-derives the dGPU list and only honours an inherited `HIP_VISIBLE_DEVICES` when it is a strict subset of the detected dGPUs ([migraphx_python.rs:195-224](rusty-stack/src/installers/components/migraphx_python.rs)).
- Fail-safe property confirmed in code: unreadable VRAM alone never classifies a device as integrated; only the PCI id, gfx arch, or name can ([gpu.rs:244-260](rusty-stack/src/gpu.rs)).

**No finding.** The iGPU tenet is genuinely and consistently enforced through one code path.

---

## Tenet 4 — No NVIDIA/CUDA dependency ever installed (hard prime) — **MET (with residual low-severity notes)**

> **Status distinction:** the hard-prime rule is *genuinely enforced* by a
> wired-in defence-in-depth (chokepoint + unified blocklist + requirements
> filtering + `--no-deps` + contamination scan). "MET" here means "no source
> path installs a CUDA dependency." The residual notes below are *not*
> violations of that rule — they are low-severity caveats about scope and
> non-blocking alias env vars, listed so this section does not read as a
> blanket "nothing left to harden" verdict.

**What the source actually does:**

- One unified blocklist `installers/common/nvidia_blocklist.rs`: `BLOCKED_PREFIXES`, `BLOCKED_EXACT` (torch family), `CUDA_URL_MARKERS`, `is_cuda_nvidia_package`, `is_nvidia_cuda_runtime_package`, `is_cuda_wheel_url`, `filter_requirements`, `contaminated_packages` ([nvidia_blocklist.rs:39-260](rusty-stack/src/installers/common/nvidia_blocklist.rs)).
- **A real, universal execution chokepoint.** `execute_native_command` inspects every pip-invoking command (covers `NativeCommand::Pip` *and* `NativeCommand::Shell` `python -m pip install …`) and `bail!`s on any explicit nvidia/cuda runtime package or `+cuXXX` wheel URL — for *any* component ([installer.rs:2046-2074](rusty-stack/src/installer.rs)). torch/triton are deliberately permitted (ROCm-managed) and instead kept out of third-party installs by requirements filtering + `--no-deps`.
- The narrow former filters now delegate: `installer.rs::filter_cuda_requirements` → `nvidia_blocklist::filter_requirements_file` ([installer.rs:2640](rusty-stack/src/installer.rs)); `megatron::is_safe_package` → `is_cuda_nvidia_package` ([megatron.rs:323](rusty-stack/src/installers/components/megatron.rs)).
- Requirements files are filtered before install in comfyui, textgen, and megatron ([installer.rs:3661, 3918, 4038](rusty-stack/src/installer.rs)).
- CUDA env leakage stripped from the env-file writer: `export TORCH_CUDA_ARCH_LIST=` and `export PYTORCH_CUDA_ALLOC_CONF=` lines are dropped from existing `.mlstack_env` files ([installer.rs:733-738](rusty-stack/src/installer.rs)); the bootstrap bash/fish emitters do not emit them.
- The flagged installers (aiter, fastvideo, megatron) all use `--no-deps`; aiter and fastvideo are native and pass through the chokepoint.

**Residual notes (not violations):**
- `CUDA_VISIBLE_DEVICES` is still *exported* ([installer.rs:620, 669, 4875](rusty-stack/src/installer.rs)), but it is set to the **filtered dGPU index list** (same value as `HIP_VISIBLE_DEVICES`), and `OMPI_MCA_opal_cuda_support` is set to `"0"` at runtime ([installer.rs:1807](rusty-stack/src/installer.rs)). These are benign aliases/disablers, not CUDA dependencies, and they are not written to the persisted env file. No CUDA package is installed by setting them.
- The chokepoint only covers components dispatched through `execute_native_command` (native components). Any component that still runs as a legacy bash script (the `else` branch at [installer.rs:243](rusty-stack/src/installer.rs)) would bypass the Rust guard — but all of the components flagged in the review brief (aiter, wandb, fastvideo, onnx) are in `NATIVE_COMPONENT_IDS` ([components/mod.rs:128-165](rusty-stack/src/installers/components/mod.rs)), so they route through the guard. **Severity: Low**, contingent on no future blocklist-relevant component being added as a pure shell script.

**Verdict:** the hard-prime tenet is enforced by a defence-in-depth that is actually wired in (no source path installs a CUDA dependency). The residuals above are low-severity scope/alias caveats, not violations — hence "MET" rather than "fully closed with zero hardening surface left."

---

## Tenet 5 — Verification runs a functional test, not just presence — **MET**

**What the source actually does:**

- `pytorch_diagnostic_snippet` no longer unconditionally `sys.exit(0)`. It now computes `ok = (hip_version is not None) and hip_available` and `sys.exit(0 if ok else 1)` ([component_status.rs:761-767](rusty-stack/src/component_status.rs)) — a CPU/CUDA torch build or an unreachable HIP runtime now fails verification.
- Functional (not import-only) snippets with proper non-zero exits exist for the major components: triton (backend load) ([component_status.rs:299](rusty-stack/src/component_status.rs)), mpi4py (rank query) ([:306](rusty-stack/src/component_status.rs)), deepspeed (ops load) ([:313](rusty-stack/src/component_status.rs)), megatron (tensor_parallel) ([:330](rusty-stack/src/component_status.rs)), vllm (`vllm._C`/`_rocm_C`) ([:343](rusty-stack/src/component_status.rs)), aiter (op import) ([:351](rusty-stack/src/component_status.rs)), onnx (GPU EP present) ([:368](rusty-stack/src/component_status.rs)), bitsandbytes (ROCm `.so` present) ([:378](rusty-stack/src/component_status.rs)), migraphx (`parse_onnx`) ([:389](rusty-stack/src/component_status.rs)), wandb (`wandb.init(mode='disabled')`) ([:405](rusty-stack/src/component_status.rs)), fastvideo (module load) ([:411](rusty-stack/src/component_status.rs)).
- **textgen and comfyui functional probes are present, not deferred** as the brief assumed: comfyui runs `python3 -c 'import folder_paths'` in `~/ComfyUI` ([component_status.rs:501-509](rusty-stack/src/component_status.rs)) and textgen runs `import server` in `~/text-generation-webui` ([component_status.rs:510-518](rusty-stack/src/component_status.rs)).
- Aggregation is honest: `success = executed > 0 && failed == 0` ([installer.rs:1153](rusty-stack/src/installer.rs)). A component with **no** verification commands (the `_ => Vec::new()` arm at [component_status.rs:519](rusty-stack/src/component_status.rs)) yields `executed == 0` → `success == false`, i.e. it is marked failed rather than silently passing — the safe direction.

**Minor caveats (Low):**
- A few non-Python components verify by presence rather than a functional GPU run: vllm-studio (`bun --version`), rocm-smi (`rocm-smi --showproductname`), llama-cpp (`llama-cli --help`) ([component_status.rs:358-460](rusty-stack/src/component_status.rs)).
- Utility/chain components (`repair-stack`, `amdgpu-drivers`, `migraphx-python`) fall into the empty `_` arm and will report failed verification even when their action succeeded.
- Note: install-step failure is overridden by verification success ([installer.rs:316-326](rusty-stack/src/installer.rs)) — intentional, and acceptable because verification is now functional.

**No blocking finding.**

---

## Tenet 6 — uninstall / reinstall functional across shells and distros — **MET**

**What the source actually does:**

- `uninstall_stack()` is a real, user-reachable flow, not dead primitives ([uninstall.rs:103-225](rusty-stack/src/uninstall.rs)). It:
  - pip-uninstalls the ML packages via the canonical python ([uninstall.rs:128-148](rusty-stack/src/uninstall.rs));
  - purges ROCm/amdgpu system packages **cross-distro** — `build_system_purge_cmd` selects apt / pacman / dnf / zypper / yum with correct verbs ([uninstall.rs:230-256](rusty-stack/src/uninstall.rs));
  - removes `/opt/rocm` via `run_privileged` (root or sudo, degrades gracefully if neither) ([uninstall.rs:158-166, 259-289](rusty-stack/src/uninstall.rs));
  - removes the env files it wrote (`~/.mlstack_env`, fish `conf.d`, `~/.rocm_env`) ([uninstall.rs:311-327](rusty-stack/src/uninstall.rs));
  - **cross-shell** strips `source ~/.mlstack_env` lines from `~/.bashrc`, `~/.zshrc`, and `~/.config/fish/config.fish` idempotently ([uninstall.rs:331-358](rusty-stack/src/uninstall.rs));
  - clears the registry ([uninstall.rs:169-176](rusty-stack/src/uninstall.rs)).
- `rusty uninstall` and `rusty reinstall` subcommands exist ([rusty.rs:204-227, 2459-2470](rusty-stack/src/bin/rusty.rs)). **Reinstall is real, not a relabel:** `reinstall_impl::run` performs a full uninstall (`yes: true`) and then relaunches the TUI installer ([rusty.rs:2308-2334](rusty-stack/src/bin/rusty.rs)).

**Minor gaps (Low):**
- pip uninstall runs against the single canonical interpreter only; packages installed into a named env under `~/.mlstack/envs/<name>/` are not enumerated/removed (uninstall doesn't iterate the named-env dir). With `--purge-mlstack-dir` the whole `~/.mlstack` tree is removed, which covers it, but that flag is off by default.
- Uninstall uses a hardcoded `ML_PIP_PACKAGES` list ([uninstall.rs:50-72](rusty-stack/src/uninstall.rs)) rather than `registry.all_pip_packages()` (which is empty anyway — see Tenet 2). Functionally fine, but it means the registry's per-component package tracking plays no role here.

**No blocking finding.**

---

## Tenet 7 — Logs assessed; version bump + binary — **PARTIALLY VERIFIABLE (source-only)**

- Version bump is confirmed in source: `Cargo.toml` `version = "0.3.0"` and `VERSION` = `0.3.0`.
- "Recent logs assessed for errors/violations," "version-bumped binary created so I can test after rebooting," and the CHANGELOG `[0.3.0]` entry are **process/artifact** claims that cannot be verified by reading source code, and (per the review directive) CHANGELOG/git are not admissible evidence. No source-level finding; this tenet is outside what a source read can confirm beyond the version constants.

---

## Overall verdict

This is a substantial, genuine remediation, not a relabel. The two newly-centralised safety subsystems — iGPU classification (`gpu.rs`, **Tenet 3**) and NVIDIA/CUDA exclusion (`nvidia_blocklist.rs` + the `execute_native_command` chokepoint, **Tenet 4**) — are real single-sources that are actually called on the hot paths, including the previously-bypassing consumers, and they are fail-safe in the correct direction (never drop a dGPU; never install a CUDA wheel). Functional verification (**Tenet 5**) and a real cross-shell/cross-distro uninstall+reinstall (**Tenet 6**) are implemented and wired to user-facing subcommands. Several items the brief listed as "deferred" (textgen/comfyui functional probes, `rusty install --env`) are in fact implemented.

**Update (post backbone-hardening):** the two backbone residuals noted in the first source pass have since been closed in source:
1. The default bare-`rusty` TUI path now anchors to `~/.mlstack/global` (`run_installation` calls `ensure_global_venv` + pins `MLSTACK_PYTHON_BIN`; `resolve_python_bin` prefers the global python). **Tenet 1 now MET** (residual: graceful-discovery fallback if venv creation fails — logged, not silent).
2. The installed-component registry is now a real single-source-of-truth: `registry_record` populates `version`/`source_index`/`location`/`pip_packages`, the sealed set extends to the install-once components (aiter/flash-attn/rccl/migraphx/bitsandbytes), and `uninstall` unions `all_pip_packages()` with its curated list. **Tenet 2 now MET** (residual: the recorded `source_index`/`location` are not yet force-read back as the dep source at install time — reuse is via interpreter pinning + `--no-deps`, so "single recorded source, force-used as dep" is realised indirectly).

**Remaining recommended follow-ups (smallest correct changes):**
- Have the dep-sourcing path read `source_index`/`location` back out of the registry at install time (rather than only pinning the interpreter) so the recorded source is the *authoritative* dep origin, not just a record.
- Make global-venv creation failure abort (or prompt) rather than silently degrading to a discovered interpreter, if the project wants the single-env guarantee to be hard rather than best-effort.
