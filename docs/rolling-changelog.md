# Rolling Changelog — Session 54e587ea (post-v0.3.1, uncommitted)

---

## FreeToken multi-GPU rail: pynccl -> RCCL verified on 2x gfx1100 (2026-08-23)

### Why only one GPU was used
`ft serve` defaults to `--tensor-parallel-size 1`, and the GGUF expert
path is TP=1-only upstream (gemma4 loader asserts the same: k-quant
blocks cannot be inner-dim sliced without requantization).

### Fixed — pynccl builds and runs against RCCL on HIP
- Link `-lrccl`; include rccl.h + the HIP runtime header instead of the
  vendored NVIDIA-only nccl 2.27 header; accept kDLROCM DLPack tensors
  (torch on ROCm types tensors kDLROCM).
- **Platform finding**: the stock system librccl is BROKEN for
  cross-GPU collectives on this host — even torch's own
  ProcessGroupNCCL fails ("operation cannot be performed in the present
  state", abort in rccl enqueue). The stack's repaired RCCL overlay
  (~/.mlstack/components/rccl/active, PYTHONPATH sitecustomize +
  MLSTACK_RCCL_OVERLAY_*) fixes it: pynccl all_reduce verified correct
  on RX 7900 XTX + RX 7800 XT (both the direct and symmetric-memory
  buffer paths).
- Launcher (~/.mlstack/bin/ft) now strips ONLY the onnxruntime
  PYTHONPATH entry and keeps the RCCL overlay vars, so TP>1 works when
  invoked through the stack; single-GPU serving unaffected.

### Remaining for full TP=2 GGUF serving (scoped)
The dense weights + attention TP machinery work via the engine; the
GGUF k-quant expert banks need a sharding story: either dequant ->
intermediate-split -> requant per rank, or expert-parallel (each GPU
owns half the experts, all-to-all instead of all-reduce). Upstream's
gemma4 GGUF loader has the same TP=1 restriction.

---

## FreeToken GGUF serving on ROCm: Ornith-1.5-35B end-to-end (2026-08-23)

### Added — qwen35moe GGUF adapter (fork `feature/rocm`)
Native GGUF loading for the hybrid GDN/MoE architecture: config from llama.cpp
KV metadata, dense weights bf16-dequantized (norms verbatim — llama.cpp
pre-bakes the Gemma +1), experts as a new `ggml` offload-bank format (native
Q4_K/Q6_K bytes, mixed quants requantized to uniform; matched round-trip
encoder at +0.12% over the Q4_K nibble floor). Three layout facts decoded and
verified per-head against HF ground truth (cos 0.98-0.999): GDN value heads
stored de-interleaved ([even|odd]) across all v-dim tensors; ssm_a =
-exp(A_log); attention q|gate half keeps per-head interleave. MTP layer
dropped. Dedicated venv install, all kernels HIP-green.

### Fixed — four HIP kernel bugs found by the e2e loop
1. ROCm 7 64-bit shuffle masks silently excluded wave64's upper 32-lane
   segment -> Q8_1 activation quantizer corrupt -> ALL MMVQ results zero on
   gfx1100. Fixed with mask-less `__shfl_xor(width=32)`.
2. `launch_pdl` kwarg rejected by ROCm Triton even when False (crashed graph
   capture); PTX tanh/ex2 intrinsics ('f' constraint) un-compilable on
   AMDGCN -> libdevice/tl.exp2 under IS_AMD constexpr.
3. ld/st PTX cache hints in fast_index_copy (invalid 'l' constraint) ->
   __ldg/plain stores; extended the HIP API shim; functional
   host_ptr_identity probe.
4. Q4_K CPU dequant + requantizer (validated vs the GPU kernel on real
   checkpoint data).

### Verified end-to-end on 2× RX 7900 XTX (gfx1100, ROCm 7.2.4)
`ft serve` Ornith-1.5-35B-A3B (Q4_K_M GGUF): llama.cpp-parity next tokens
("The capital of France is Paris"), coherent chat + reasoning channel,
~39 tok/s decode, attention=triton + MoE=offload auto-selected, CUDA graphs
captured. Subsystem validations: 37/37 upstream attention/rotary tests, GDN
chunk/decode vs the pure-torch reference, full offload loop (decode+prefill)
at ~1.1%, multi-expert MMVQ on real banks at 0.6-1.2%.

---

## FreeToken: first-class MoE serving component on ROCm (2026-08-22)

### Added — `freetoken` component (VAL-INSTALL-050..054)
MoE-offload LLM serving engine (FlashML-org) as a first-class rusty-stack
component, built from the `scooter-lacroix/FreeToken` fork's `feature/rocm`
branch (5-commit upstreamable series: CppExtension HIP shim, tvm-ffi
`backend="hip"` rail, GGUF hipify + ROCm 7 64-bit-mask shuffle fix, HIP-aware
arch gates, README ROCm section). Deliberate divergence from the
megatron/flash-attn pattern: installs into a **dedicated venv**
(`~/.mlstack/venvs/freetoken`) because FreeToken's pins (transformers>=5.5,
numpy<2.5) are ahead of the global env's vllm/megatron set. Torch comes from
the same ROCm wheel index as the sealed core; launcher shim
`~/.mlstack/bin/ft` sanitizes PYTHONPATH (onnxruntime/RCCL shims must not
shadow venv imports); install-time HIP smoke test (backend→triton, pinned
identity, LRU slot cache) is timeout-guarded. Full registration sweep:
NATIVE_COMPONENT_IDS (42), dep closure [pytorch, rocm], TUI entry,
detection/registry (venv python + FREETOKEN_VENV_PYTHON), manifest
(0.1.2-rocm.1, Candidate), uninstall, enhanced verification, dispatch tests.
Docs: `docs/extensions/freetoken_guide.md`, `docs/guides/freetoken_amd_guide.md`.
Submodule: `Fork/FreeToken` → scooter-lacroix fork, branch `feature/rocm`.
Verified on 2× RX 7900 XTX (gfx1100, ROCm 7.2.4, torch 2.13+rocm7.2):
extensions build+import, index/store JIT kernels numerics via hipcc, GGUF
Q4_K dequant on real checkpoint data, engine auto-backend → triton, HIP
CUDA-graph capture, flashlib `lru_ensure` slot-cache invariants.

Staging log of all work this session (pre- and post-context-compaction), to be
merged into the root `CHANGELOG.md` under a new version section when the build
is prod-ready. Grouped by theme; root-cause explanations kept verbatim where
they matter. All changes compile clean (`cargo fmt --all --check`,
`cargo clippy --workspace --all-targets`, `cargo build --release` — zero
warnings, no new `#[allow]`).

Scope: **48 files changed, +3594 / −2583** across `rusty-stack/` + a one-line
fork fix in `Fork/llama.cpp-turboquant-hip` (must be pushed to `main` on
github — the installer clones `main`).

---

## Follow-up fixes (2026-06-28)

### Fixed — Rusty Llama `cmake --install` failure (unbuilt test/example binaries)
The fork defaults `LLAMA_BUILD_TESTS`, `GGML_BUILD_TESTS`, `LLAMA_TESTS_INSTALL`,
and `LLAMA_BUILD_EXAMPLES` to ON (standalone). The installer builds only the
tool targets (`llama-cli`/`llama-bench`/`llama-server`), so `cmake --install`
failed trying to install **unbuilt** `test-tokenizer-0` (test) and
`llama-batched` (example) binaries. `cmake_flags` now passes
`-DLLAMA_BUILD_TESTS=OFF -DGGML_BUILD_TESTS=OFF -DLLAMA_TESTS_INSTALL=OFF
-DLLAMA_BUILD_EXAMPLES=OFF` (+ `LLAMA_BUILD_TOOLS=ON`, `LLAMA_BUILD_SERVER=ON`
so the 3 targets exist) — install set = tools + libs only. **Proven:** build
the 3 targets + `cmake --install` → exit 0.

### Fixed — FA-Triton runtime dependency on AITER (was undeclared)
FA-Triton's Triton backend imports `aiter.ops.triton...flash_attn_triton_amd`
at runtime, but neither `get_dependencies` nor `derive_dependencies` declared
aiter, and the FA installer's `fa_deps` (einops/ninja/packaging/psutil) +
`--no-deps` don't pull it. So a FA-Triton-only install would break on import
without aiter. `flash-attn-triton` now declares `aiter` (planner installs it
first); `flash-attn-ck` does **not** (uses composable_kernel, not aiter). Test
added (`test_flash_attention_dependencies`). Note: AITER reporting "installed"
on your box was the **AITER component** running (selected), not FA-Triton
pulling it — FA-Triton just benefits from it being present.

---

## Changed — Component taxonomy & TUI honesty

- **Category rename `Verification` → `Maintenance`.** Unified across BOTH
  `Category` enums (`core/types.rs`, `app.rs`); `categories_len` stays 7.
  Repair + verify-* actions now live under Maintenance (they are actions, not
  stateful installs).
- **Component reorg.** migraphx/megatron/llama-cpp → **Core**; rocm-smi +
  pytorch-profiler → **Foundation**; repair-stack → **Maintenance**. AITER →
  **Core** (was Extension, an oversight). `ml-stack-core` removed from the TUI
  list (redundant bundle — its parts are already individual components) but its
  dispatch arm + module are kept because `repair-stack` still routes through it.
- **Experimental badge + note.** `Component` gained `experimental: bool` +
  `note: Option<String>`; the panel renders a `⚠ EXPERIMENTAL BUILD` line + the
  note. Used by Flash Attention (CK). Selectable + installable — never greyed.
- **"Coree" glyph artifact.** Dropped the U+FE0F variation selector on the Core
  gear icon so it renders as "Core" with no trailing phantom char.

## Added / Changed — Flash Attention split (Triton active / CK experimental)

- **Split into two explicit backends.** `flash-attn-triton` (Core, selected,
  full forward+backward — the only RDNA3 backend with a backward pass) and
  `flash-attn-ck` (Core, experimental, **forward-only on RDNA3** — the
  composable-kernel backward pass is not implemented,
  ROCm/composable_kernel#1434). The CK note states exactly what it
  enables/blocks (inference/generation accelerated; training/backprop falls
  back to unfused PyTorch SDPA; RDNA4 gains backward with `deterministic=False`).
- **Backend honesty via marker.** Detection is now **marker-based**: a component
  reports installed ⟺ `flash_attn` importable AND
  `~/.mlstack/flash-attention/.backend` matches (`triton`/`ck`). Deleted the
  legacy `path_exists(ml_stack/flash_attn_amd*)` false-positive surface. A
  flash-attn installed outside the TUI reports "not installed" until reinstalled
  through it (known backend > guess).
- **Installer: two fixed-backend arms** (`installer.rs`). Both clone
  `ROCm/flash-attention` into `~/.mlstack/flash-attention` (idempotent
  `git_clone_or_pull` + safe.directory + ownership fix) and
  `pip install -v --no-build-isolation --no-deps`. Triton sets
  `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`; CK omits it (ROCm CK = default
  backend). On success each writes the `.backend` marker (FATAL via `map_err` —
  no marker on pip failure). `MAX_JOBS` capped for the RAM-heavy CK build.
- **Mutual-exclusivity toggle (TUI).** Both backends install the *identical*
  `flash_attn` package, so the last-installed overwrites the other (marker
  tracks the active one). `toggle_component` now deselects the sibling FA
  backend when one is turned on; `toggle_all` keeps at most one. Triton + CK
  notes both state the mutual-exclusivity + workload tradeoff (Triton =
  training/backward; CK = forward-only inference).

## Added — Flash Attention benchmarking

- **Functional forward test** (CK, RDNA3): `flash_attn_func` +
  `flash_attn_qkvpacked_func`, fp16/bf16, causal/non-causal, seq 2048–8192,
  correctness vs torch SDPA ≤ 1.5e-2. **7/7 PASS.**
- **Genuine-model forward benchmark** wired as `flash-attention-ck`: a real
  151M-param Llama-style decoder (RMSNorm + rotary + GQA via `flash_attn_func` +
  SwiGLU), forward-only under `no_grad`. Registered in `available_benchmarks()`,
  `run_benchmark()`, `run_all()`; new `flash-attention-ck-performance`
  Performance component (experimental). Reproducible (no model download).
- **Real-model characterization (Qwen3-4B, bf16):** FA-CK vs Triton vs SDPA.
  Prefill: both FA backends beat SDPA (1.06×–1.35×, CK marginally faster at long
  context). Decode: SDPA 36.0 > CK 26.6 > Triton 2.1 tok/s (Triton cold-cache
  JIT per decode shape). Only Triton has a backward pass on RDNA3.
- **`parse_progress`** now parses ninja `[N/M]` build format (FA-CK build UX).

## Fixed — Rusty Llama (llama.cpp-turboquant-hip) fork + installer

- **CMakeCache validator false-negative (root cause of the build "failure").**
  `validate_cmake_cache` matched the prefix `GPU_TARGETS:STRING=`, but a
  typeless `-DGPU_TARGETS=` is stored as `GPU_TARGETS:UNINITIALIZED=`. A
  correctly-configured build was rejected → every source install aborted after
  configure. Fixed to parse the value regardless of cmake cache type. Regression
  test added (`test_validate_cmake_cache_accepts_uninitialized_gpu_targets`).
- **Fork multi-arch compile gate (`Fork/…/vendors/hip.h`).** The fork included
  `<rocwmma/rocwmma-version.hpp>` whenever `GGML_HIP_ROCWMMA_FATTN` was on —
  gated on the *flag*, not the *arch*. Result: (a) failed when rocwmma-dev was
  absent (every TU pulling `common.cuh`), and (b) dragged RDNA3-only WMMA code
  into the gfx1030 (RDNA2) pass. Fixed with an **arch + `__has_include` gate**:
  WMMA auto-on for RDNA3+ (gfx1100/1101/1150/1151) / RDNA4 (gfx1200/1201) when
  rocwmma is present, cleanly `#undef`'d for RDNA2 or when absent. Proven: full
  union `gfx1030;gfx1100;gfx1101` compiles (100%, exit 0); `llama-bench` on the
  7900 XTX = pp64 5921 t/s, tg32 276.5 t/s; ROCm-only linkage (no `libcuda`).
- **`gpu_targets_for_channel` keeps the full multi-arch union** (gfx1030;
  gfx1100;gfx1101[;gfx1200]) — the hip.h gate is what makes it compile. Do NOT
  narrow to a single arch family.
- **Re-run idempotency.** `mkdir_build_dir` now clears any stale clone/build
  before re-cloning (`rm -rf && mkdir -p`): git refuses to clone into a non-empty
  dir and a partial `CMakeCache.txt` would shadow fresh `-D` flags.
- **Purged hardcoded dev path.** `verify_installed_binary` + the installer arm no
  longer reference `~/Documents/…/Fork`; verification resolves binaries from the
  managed install dir + searches `~/.mlstack/models` + the clone dir for a
  verification model (best-effort).
- **Private-repo auth.** `resolve_auth_token()` chain: runtime
  `GITHUB_INSTALLER_TOKEN` → build-time embedded `LLAMA_CPP_DEPLOY_TOKEN`
  (`option_env!`) → none. Token is injected at build time, never committed; scope
  read-only to the single repo (still recoverable via `strings` — inherent to
  client-side secrets).
- **Label fix:** llama.cpp component renamed "llama.cpp (HIP/ROCm)" →
  **"Rusty Llama (llama.cpp)"**.

## Changed — Unified ROCm environment (iGPU never visible)

Three drifted env writers (`bootstrap/env_setup.rs`, `permanent_env.rs`,
`installer::ensure_mlstack_env`) consolidated into **one canonical generator**.

- **Canonical generator** (`bootstrap/env_setup.rs`): bash + fish from one
  source. Added the authoritative **`ROCR_VISIBLE_DEVICES`** (the ROCr/HSA
  runtime filter — the one that actually hides the iGPU; `HIP_VISIBLE_DEVICES`
  alone was insufficient) to both. Added the missing vars so it's a true
  superset: `ROCM_PATH`, `HIP_PATH`, `GPU_ARCHS`, `PYTORCH_ROCM_ARCH`,
  `AMDGPU_ASIC_ID_TABLE_PATH/_PATHS`, `CUDA_HOME`. Device-filter vars are
  **unconditional** (managed — a stale override can never re-expose the iGPU).
- **`ensure_mlstack_env` delegates** to the canonical generator (no more inline
  `format!`), writes `~/.mlstack_env` **and** the fish
  `~/.config/fish/conf.d/mlstack_env.fish` (auto-loaded by fish). Keeps live
  hw-detect (`gpu::detect_discrete_amd_gpus`) + `sanitize_mlstack_env`.
- **`permanent_env::generate_env_file_content` delegates** to canonical (the dead
  duplicate is retired).
- **Auto-source on shell launch (Global install).** New
  `offer_mlstack_env_source()` idempotently appends
  `[ -f "$HOME/.mlstack_env" ] && source "$HOME/.mlstack_env"  # mlstack-rocm-env`
  to `~/.bashrc` + `~/.zshrc` (fish needs nothing — conf.d auto-loads). Called
  post-install, gated on the global-install flag (named/isolated envs are NOT
  auto-sourced — stated in the TUI Permanent-Env note). The marker matches
  `uninstall::strip_shell_sourcing`, so uninstall cleans it.
- **verify respects the env.** `verify_installed_binary` runs `llama-cli` /
  `llama-bench` / `rdna3` via `bash -c 'source ~/.mlstack_env; exec …'` so ROCm
  ops never enumerate the iGPU regardless of how rusty-stack was launched.
- **Proven end-to-end:** fresh `bash` + `fish` shells auto-source
  `ROCR_VISIBLE_DEVICES=0,1`; `llama-bench` via the verify path sees 2 dGPUs (no
  gfx1036 APU) vs 3 devices unsourced. The APU-probe segfault is gone.

## Added — ROCm installer

- **rocwmma** added to `pacman_rocm_packages()` (stable/latest) — the WMMA
  flash-attention backend dep used by Rusty Llama's `fattn-wmma-f16`. apt/dnf/
  zypper already pull it via the `rocm-libs` meta-package.

## Fixed — Installer hardening / dead-code purge

- **All 13 `#[allow]` suppressions eliminated** — each resolved at the root
  (dead code deleted, latent bugs wired, stale allows removed), not blindly
  killed.
- **Dry-run wiring:** AMDGPU driver installer (`build_package_install_command`)
  + Repair (`run_repair_sequence`) honor `dry_run` (echo-preview / skip-steps).
- **Dead code removed:** `fix_env_assignment`, `execute_command_sequence`,
  `VerificationResult::Warning`, `RecordingSmokeTester`, orphaned
  `AtomicBool`/`Ordering` imports, `NativeInstallerContext.input_rx`,
  `RustAdapter.component_id`/`with_id`, `flash_attention_ck.rs` legacy cmake
  module (88 lines), `migraphx_python.rs`, `vllm_multi.rs`, `pytorch_profiler.rs`,
  `wandb.rs`.
- **MiGraphX:** post-install lib-load check via `migraphx_driver_status()`;
  multi-modal Python→driver fallback verification.

## Docs

- New review-prompt variant
  `docs/review-prompts/v0.3.2-core-reorg-flash-attention-verification-source-only.md`.

---

## Outstanding / follow-ups (not in this log's scope)

- **Push the fork `hip.h` gate to `main`** on github — the installer clones
  `main` (`DEFAULT_BRANCH` in `installers/components/llama_cpp.rs`); without it
  the next install re-clones the unfixed repo. (Fix is on
  `feature/upstream-sync-2026-05`; merge to `main`.) This is a tracked
  follow-up in the fork repo (`scooter-lacroix/llama.cpp-turboquant-hip`), not
  a defect in this PR's code: the installer's clone/pin contract is correct,
  it simply needs the upstream fork's `main` to carry the fix. Until merged,
  the submodule pin (`Fork/llama.cpp-turboquant-hip @ c39871c9c`) is the
  authoritative tested SHA.
- **VerificationCommand execution layer should source `~/.mlstack_env`** (the
  14th shallow-verification instance): FA verify failed on a root-owned
  `~/.triton/cache` because the verify ran without `TRITON_CACHE_DIR`. Symptom
  fixed (`sudo chown -R scooter:scooter ~/.triton`) but the code root cause
  remains — generalize the `run_rocm_tool_with_env` pattern to the verification
  runner.
- Installer python-routing audit (`resolve_python_bin` + all
  `--break-system-packages` usage → confirm managed venv is always the target).
- Amend commits (drop Co-Authored-By) + force-push PR #22.
