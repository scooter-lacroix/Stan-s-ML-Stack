# FastVideo Remediation — Task List

**Target:** `rusty-stack/src/installers/components/fastvideo.rs`, shared dependency ownership, dispatch, and verification
**Fork:** `scooter-lacroix/FastVideo@22e448771ebf5c81108f1c57b9c7ef4d5c26d182` (`feature/rocm-native-no-cutlass`)
**HW target:** 7900 XTX (gfx1100) + 7800 XT (gfx1101); APU gfx1036 excluded by Rusty visibility
**Constraints:** FastVideo consumes only Rusty-managed dependencies; local installs are offline, `--no-build-isolation`, and `--no-deps`; no NVIDIA/CUDA packages or source; no `cargo test`; zero warnings; CPU-limited builds (`nice -n 19 -j 2`).

## Decisions (locked)
- **Build scope:** both GPUs — `CMAKE_HIP_ARCHITECTURES=gfx1100;gfx1101`.
- **Flash-attn:** route to env `flash_attn` — prefer **CK** backend when installed (RDNA3 forward-pass-capable), else **Triton**. The current marker on this box is **CK**. Replace the patched-out bundled CK.

---

## Phase 1 — Mechanical installer fixes (definite; unblocks install)

- [x] **1.1 Immutable source:** remove any previous build tree, clone fresh, and checkout detached at `22e448771ebf5c81108f1c57b9c7ef4d5c26d182`; no submodules or install-time patching.
- [x] **1.2 Multi-arch build:** derive selected dGPU architectures and pass `CMAKE_HIP_ARCHITECTURES=gfx1100;gfx1101`.
- [x] **1.3 Source policy:** run isolated Python (`-I -S`), prove exact HEAD and an empty worktree including untracked files, then reject gitlinks, `.gitmodules`, CUTLASS/CuTe/ThunderKittens, CUDA/NVIDIA metadata, and unexpected kernel sources.
- [x] **1.4 Dependency isolation:** remove FastVideo `BuildDeps`; preflight exact Rusty-managed prerequisites and install kernel/root with network and dependency resolution disabled.
- [x] **1.5 Contamination guard:** snapshot distributions immediately before install, allow only `fastvideo`/`fastvideo-kernel` changes, and reject packages using Rusty’s canonical NVIDIA runtime-prefix policy.
- [x] **1.6 Central ownership:** PyTorch provisioning owns exact build/runtime prerequisites; FastVideo only validates them read-only.

## Phase 2 — Functional verification (real, not import-only)

- [x] **2.1 Smoke test implementation:** isolate **each physical ROCr visibility token** in its own child process, clear conflicting HIP/CUDA/legacy AMD masks, and run `fastvideo_kernel.int8_quant` on the child's logical `cuda:0`, synchronizing and checking dtype, shape, finite scale, and GPU identity. Both component-specific and aggregate verification paths use the same snippet.
  - Runtime execution passed on gfx1100 + gfx1101.

## Phase 3 — Flash-attn routing (needs source investigation first)

- [x] **3.1 Source study:** pinned fork `22e448771ebf5c81108f1c57b9c7ef4d5c26d182` imports the environment package's public and private `flash_attn` APIs from Python. Bundled FlashAttention C++ is removed.
- [x] **3.2 Backend selection:** the installer preflight reads `~/.mlstack/flash-attention/.backend`, accepts exactly `ck` or `triton`, and imports the required public/private APIs. Current backend: **CK**.
- [x] **3.3 User installation/runtime validation:** Rusty installed the root package dependency-free against managed `flash_attn`; the compiled-op smoke passed on both dGPUs.

---

## Build / deploy
- [x] Exact staged tree: 23 FastVideo + 3 managed-owner-focused tests pass; `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` are clean.
- [x] Exact staged tree: `nice -n 19 cargo build --release -j 2` passes.
- [x] Code commit: `5beed1c` (`fix(rusty-stack): harden FastVideo ROCm install`).
- [x] Atomic binary replacement complete; source/deployed SHA-256: `18007be04bd194b94562220532ac6c983aef1319aacb9b694fe9469099ac2130` (`rusty-stack 0.3.1`).

## Validated on target system
- [x] FastVideo installed through the replaced Rusty binary.
- [x] Installer built both `gfx1100` and `gfx1101`.
- [x] Shared compiled-op smoke passed on 7900 XTX + 7800 XT.
- [x] FlashAttention routed to the Rusty-managed CK package.
- [x] Post-install distribution guard reported no dependency changes or NVIDIA/CUDA packages.

---

## Standing / lower priority (not FastVideo)
- [ ] Run `vllm-performance` benchmark (test engine-init + flip "Install and run benchmarks" label).
- [ ] Verify-without-env bug: `VerificationCommand` runs without sourcing `~/.mlstack_env`.
- [ ] Installer python-routing audit (env-torch enforcement across components).
- [ ] Amend commits (remove `Co-Authored-By`), force-push PR #22.
- [x] 2026-08-26: torchao 0.15.0 emits cpp-extension skip warning under managed torch 2.13.0+rocm7.2 — logged, cosmetic, no functional impact on fastvideo_kernel ops.

---

## Upstream sync — fork rebased onto hao-ai-lab/FastVideo main (2026-08-26, same day)

Divergence: merge-base `4105094f` (2026-03-13) → upstream moved **356 commits** (2,186 files; fastvideo-kernel expansion, Blackwell sm100a kernels, model zoo, packaging churn), while the fork carried only **13 ROCm commits**. Rebase executed in scratch by mapping agent; all 13 replayed linearly, 9 conflict stops resolved delete-wins (bundled CUTLASS GEMM, H100 .cu pair, `.gitmodules`, publish workflow), plus a fixup stripping policy violators newly added upstream (vbench gitlink, attn_qat CuTe bundle). One more fixup integrated locally after the *runtime* scanner caught three unreferenced Blackwell sources (`block_sparse_*_sm100a.cu`, `ulysses_all_to_all.cu`) outside the audited five-file native allowlist.

**New audited pin: `1d277a9b03a86abba53d08e0969117e1b7466f61`** (branch `rocm-sync-candidate` in Fork/FastVideo; safety tag `pre-sync-fork-tip` = old `22e448771`). Bundle + full matrix: `/tmp/fastvideo_sync/MAPPING_REPORT.md`.

- [x] Policy suite 41/41 (agent) **+ runtime source-policy gate green on real clone** (E2E).
- [x] HIP wheel rebuild on synced tree (gfx1100;gfx1101 via new `PYTORCH_ROCM_ARCH` env pin) + both-GPU E2E replay; GPU1 functional verify `int8_quant=ok` (GPU0 deferred — resource directive).
- [x] Builder hardening discovered during sync: torch's LoadHIP.cmake silently overrides `CMAKE_HIP_ARCHITECTURES` via `rocm_agent_enumerator`; installer now exports `PYTORCH_ROCM_ARCH=<arch list>` alongside CMAKE_ARGS (test-enforced).
- [x] Publishing (operator, optional): local submodule holds `rocm-sync-candidate`; making the remote branch tip means a non-fast-forward push (`git push -f fork rocm-sync-candidate:feature/rocm-native-no-cutlass`) or keeping it as its own branch — either is safe; pre-sync state tagged.
- Note: an unlinked-CWD under child processes (deleting `/tmp/FastVideo_ROCm_build` from inside itself before running verify) reproduces the Intel OpenMP SIGABRT deterministically on this torch wheel — harness artifact only; production runner cwd never sits inside the build dir. Replay scripts now `cd /home/scooter` before cleanup.

---

## Incident D — install rejection + ABI-stale smoke (2026-08-26, resolved same day)

**Chain of evidence** (`~/.mlstack/logs/rusty-stack.log`):
1. vLLM reinstall @12:52 downgraded setuptools 81.0.0 → 79.0.1 (its dependency set pins `>=77.0.3,<80.0.0`; log lines 40087→40130).
2. Off-log month-gap drift also moved diffusers 0.33.1 → 0.37.1 and wcwidth 0.2.13 → 0.8.2, and torch 2.12.1+rocm7.2 → 2.13.0+rocm7.2.
3. FastVideo's read-only pin preflight correctly refused to touch the mutated env (log 40279–40282).
4. Design smell: automatic verification ran anyway after the failed install, launched per-GPU child processes against the stale Jul-19-built `fastvideo_kernel` (built for torch 2.12 ABI) → `turbodiffusion_ops.int8_quant` bound `quant_cuda=None` → `TypeError: 'NoneType' object is not callable` plus torch weakref-exit noise (log 40283–40307).

**Root causes were structural, not environmental**: the 11 managed pins existed as two hand-copied literal lists (owner registry vs consumer copy), provisioning only ran inside the sealed-gated pytorch arm, unpinned sibling installs clobbered the pins afterward, and verification executed unconditionally even on install failure.

### Resolution
- [x] **Single-source the pins** — `fastvideo.rs::managed_dependencies_preflight` renders from `managed_python_build_tools::MANAGED_PYTHON_PACKAGES`; duplicate literals deleted; tests assert exact-once rendering and byte-equality with the registry.
- [x] **Co-existence pin** — canonical setuptools is **79.0.1**, satisfying vLLM's `<80` window *and* every ROCm build backend; ends the perpetual downgrade oscillation. diffusers/wcwidth restored to owned pins on-box (pip `--no-deps`), confirming ownership enforcement.
- [x] **Self-healing reconciliation** — `installer.rs::reconciled_managed_python_packages` runs read-only verify after every successful non-PyTorch component completion; on drift it reinstalls the exact owned set resolver-free and logs a loud `[managed-deps]` verdict. Enforcement lives with Rusty Stack systems, not operator memory.
- [x] **Smoke gating** — verification no longer executes when the install step failed; verdict reports "Verification skipped" honestly instead of launching GPU children against stale artifacts.
- [x] **Shipped vendored fork** — `Fork/FastVideo` git submodule pinned at `22e448771ebf5c81108f1c57b9c7ef4d5c26d182`; installer resolves source as `MLSTACK_FASTVIDEO_SOURCE` env override (must match pinned commit) → nearest ancestor `Fork/FastVideo` at that commit → GitHub URL fallback. Local clones pass `-c protocol.file.allow=always`. The immutable-source policy still verifies the tree before compile.
- [x] Incidental: repaired pre-existing syntax error `migraphx_python.rs:794` (assert closing tokens swallowed by trailing comment — branch did not compile) and two new-toolchain clippy `useless_format` denials in `bootstrap/env_setup.rs` (brace-unescaping made explicit).

### Verification evidence
- [x] Rebuild green: `cargo fmt --check` clean, `clippy --all-targets -- -D warnings` exit 0, full lib suite **1602 passed / 0 failed / 8 ignored**, release build ok (`nice -n19 -j2`).
- [x] Binary replaced atomically three times during the session; **deployed fingerprint: SHA-256 `c4c59f7e42cae746…`** (`~/.cargo/bin/rusty-stack`, still version string 0.3.2); rollback backups kept as `~/.cargo/bin/rusty-stack.bak-<sha8>`.
- [x] Full pipeline replayed against production bytes (guard scripts dumped from the Rust builders via `RUSTY_DUMP_FASTVIDEO_DIR`, test `dumps_pipeline_scripts_when_ops_env_requests_it`): clone-from-vendored fork → managed-pin preflight ✅ → CK flash backend ✅ → immutable source policy ✅ → snapshot guards ✅ → **fastvideo_kernel rebuilt against live torch 2.13 ABI** (`.so` mtime verified changed) → root package ✅ → contamination guard ✅.
- [x] Functional verification snippet (production bytes) on **physical GPU1**: `[FastVideo] device=0 name=AMD Radeon RX 7800 XT int8_quant=ok shape=(2,128) dtype=torch.int8 scale_finite=True`. GPU0 intentionally not probed (resource directive: owned by another agent this session); re-run both-GPU verify when idle.

### Performance probe (GPU1, fp16→int8, 300 iters after warmup, shared-box caveat)
| config | (2,128) | (4,1024) | (16,4096) |
|---|---|---|---|
| default | 6.7µs | 6.5µs | 6.9µs |
| PYTORCH_CUDA_ALLOC_CONF=cudaMallocAsync | 22.6µs ❌ | 22.2µs ❌ | 22.5µs ❌ |
| HSA_ENABLE_SDMA=0 | 6.7µs | 6.6µs | 7.0µs |

Op is kernel-launch-bound; **defaults are already optimal** — do not enable `cudaMallocAsync` for this workload (3.3× regression).

### Additional findings fixed on the way (2026-08-26)
- [x] **pip same-version no-op bug (structural!)** — even a successful install would silently keep an old-ABI kernel because `pip install .` skips equal versions. `--force-reinstall` added to both builders; tests now require it.
- [x] **Verification children OpenMP collision** — torch 2.13.0+rocm bundles Intel OpenMP; importing fastvideo (loads aiter's libgomp) before CUDA init SIGABRTs with `recursive_init_error`. Snippet child env now carries `KMP_DUPLICATE_LIB_OK=TRUE` by default (scoped to FastVideo verification only).
- [x] **User-site shadow trap (operator)** — interactive `pip install` under the uv cpython resolved to `~/.local/lib/python3.12/site-packages`, leaving global drift "repaired" that wasn't. Shadow copies purged; pins force-reinstalled into global and origin-verified via `importlib.metadata.distributions()`.
- [ ] Standing follow-ups: identify duplicate OpenMP provider properly (retire KMP band-aid); dual-GPU re-verify when box idle; corrupt `~yping-extensions` dist-info dir in global site-packages awaits cleanup.
