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

- [x] **2.1 Smoke test implementation:** initialize FastVideo and run `fastvideo_kernel.int8_quant` on **each visible logical GPU**, synchronizing and checking dtype, shape, and finite scale. Both component-specific and aggregate verification paths use the same snippet.
  - Runtime execution on gfx1100 + gfx1101 remains in **Validate (user runs)** below.

## Phase 3 — Flash-attn routing (needs source investigation first)

- [x] **3.1 Source study:** pinned fork `22e448771ebf5c81108f1c57b9c7ef4d5c26d182` imports the environment package's public and private `flash_attn` APIs from Python. Bundled FlashAttention C++ is removed.
- [x] **3.2 Backend selection:** the installer preflight reads `~/.mlstack/flash-attention/.backend`, accepts exactly `ck` or `triton`, and imports the required public/private APIs. Current backend: **CK**.
- [ ] **3.3 User installation/runtime validation:** Rusty will install the root package dependency-free against managed `flash_attn`. The user will run the actual Rusty installer and compiled-op smoke on both dGPUs.

---

## Build / deploy
- [x] Exact staged tree: 23 FastVideo + 3 managed-owner focused tests pass; `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` are clean.
- [x] Exact staged tree: `nice -n 19 cargo build --release -j 2` passes.
- [x] Code commit: `5beed1c` (`fix(rusty-stack): harden FastVideo ROCm install`).
- [x] Atomic binary replacement complete; source/deployed SHA-256: `18007be04bd194b94562220532ac6c983aef1319aacb9b694fe9469099ac2130` (`rusty-stack 0.3.1`).

## Validate (user runs)
- [ ] Install FastVideo through the replaced Rusty binary.
- [ ] Confirm the installer builds both `gfx1100` and `gfx1101`.
- [ ] Confirm the shared compiled-op smoke passes on 7900 XTX + 7800 XT.
- [ ] Confirm FlashAttention routes to the Rusty-managed CK/Triton package.
- [ ] Confirm the post-install distribution guard reports no dependency changes or NVIDIA/CUDA packages.

---

## Standing / lower priority (not FastVideo)
- [ ] Run `vllm-performance` benchmark (test engine-init + flip "Install and run benchmarks" label).
- [ ] Verify-without-env bug: `VerificationCommand` runs without sourcing `~/.mlstack_env`.
- [ ] Installer python-routing audit (env-torch enforcement across components).
- [ ] Amend commits (remove `Co-Authored-By`), force-push PR #22.
