# Rusty Llama Hardening & Major Version Release — Specification Bible

> **AUTHORITY**: This document is the single source of truth for all implementation work in this scope. No deviation without explicit owner approval. All agents MUST read this before touching code.

**Date**: 2026-05-30
**Version**: 1.0
**Owner**: scooter (Stan)

---

## 1. Scope

Three workstreams:

1. **CUDA Isolation Guards for Rusty Llama** — Extend Rusty Stack's Nvidia/CUDA contamination defenses to cover the llama.cpp-turboquant-hip build and install pipeline end-to-end.
2. **Upstream Sync & RDNA Optimization** — Add upstream llama.cpp as a git remote, cherry-pick or merge relevant ROCm/HIP/RDNA3 improvements while preserving local RDNA2/3/4 work.
3. **Docs & Changelog for Major Version** — Update all documentation, CHANGELOG, README, and highlight Rusty Llama + Windows support as key features of the Anagami release.

---

## 2. Architecture Context

### 2.1 Codebase Layout (Relevant Files)

```
rusty-stack/
  src/
    installer.rs                     # Main orchestrator (6963 lines)
    installers/
      common/
        guard.rs                     # InstallerError enum, NvidiaContamination variant
        env_validation.rs            # .mlstack_env validation
        rocm_env.rs                  # ROCm detection facade
      components/
        llama_cpp.rs                 # Rusty Llama installer (~900 lines)
        mod.rs                       # NATIVE_COMPONENT_IDS, is_native_component(), topological_sort()
    platform/
      linux.rs                       # ROCm path/version detection
      windows.rs                     # Windows platform support (528 lines)
      wsl.rs                         # WSL2 detection (641 lines)

Fork/
  llama.cpp-turboquant-hip/          # Rusty Llama C++ fork
    CMakeLists.txt                   # Top-level cmake (GGML_HIP=ON, GGML_CUDA exists but OFF)
    ggml/
      CMakeLists.txt                 # ggml backend cmake (GGML_CUDA option defined, defaults OFF)
      src/ggml-hip/                  # RDNA3 WMMA, probes, validation .cu files
    scripts/
      build_rdna2_llama.sh           # RDNA2 build script
      validate_hygiene.sh            # Pipeline hygiene validator
    build/CMakeCache.txt             # Current build: GGML_HIP=ON, GGML_CUDA=OFF
```

### 2.2 Current CUDA Guard Gaps in Rusty Llama Path

| Guard | pip-based installers | llama_cpp.rs |
|-------|---------------------|--------------|
| `filter_cuda_requirements()` | ✅ Used by Megatron, vLLM | ❌ N/A (CMake, not pip) |
| `NvidiaContamination` error | ✅ Defined in guard.rs | ❌ Never raised by llama_cpp.rs |
| Explicit `-DGGML_CUDA=OFF` | N/A | ❌ Not set — relies on default |
| Post-build binary linkage check | N/A | ⚠️ `has_rocm_linkage()` exists but not enforced |
| Pre-build CUDA toolkit detection & abort | N/A | ❌ Missing |
| CMake cache validation | N/A | ❌ Missing |

### 2.3 Upstream llama.cpp HIP Changes to Merge

Key upstream commits since fork divergence:

| Commit | Description | Impact |
|--------|-------------|--------|
| `db9d8aa` (Mar 22, 2026) | Native BF16 flash attention for vec kernel | Performance uplift for RDNA3 |
| `d6f3030` (Apr 9, 2026) | Backend-agnostic tensor parallelism | Multi-GPU improvement |
| `3408072` (Mar 19, 2026) | Avoid compiler bug in RDNA code gen on Windows debug | Windows stability |
| `d63aa39` (Mar 12, 2026) | Compile debug builds with -O2 on HIP | Build reliability |
| `b49d8b8` (Mar 19, 2026) | CI: HIP quality check | CI alignment |
| `80d28f1` (Oct 27, 2025) | Fix AMDGPU_TARGETS, update docs | Build correctness |

---

## 3. Requirements

### R1: CUDA Isolation in llama_cpp.rs

**R1.1**: `cmake_flags()` MUST explicitly include `-DGGML_CUDA=OFF` in all channel configurations. This is a hard guard — not a default-reliance.

**R1.2**: Before invoking cmake, the installer MUST check for CUDA toolkit presence (`nvcc`, `nvidia-smi`, `/usr/local/cuda`) and emit a `log_warn()` if detected, noting that CUDA will NOT be used. This is informational, not blocking — some users may have both installed.

**R1.3**: After a successful build, the installer MUST call `has_rocm_linkage()` on the installed binary and verify it returns `true`. If linkage check fails, the install MUST be marked as failed with a clear error message.

**R1.4**: After a successful prebuilt binary download, the installer MUST also run `has_rocm_linkage()` verification.

**R1.5**: Add a new function `validate_cmake_cache()` that, after cmake configure, reads the generated `CMakeCache.txt` and asserts:
- `GGML_HIP:BOOL=ON`
- `GGML_CUDA:BOOL=OFF`
- `GPU_TARGETS` matches the expected value from `cmake_flags()`

If any assertion fails, abort the build with a descriptive error.

**R1.6**: The `cmake_flags()` function MUST also set `-DGGML_VULKAN=OFF` and `-DGGML_METAL=OFF` to prevent accidental backend contamination on systems with those SDKs present.

**R1.7**: All new guard logic MUST have unit tests with the `VAL-GUARD-*` validation assertion prefix.

### R2: Upstream Sync

**R2.1**: Add `upstream` remote pointing to `https://github.com/ggml-org/llama.cpp.git` in the Fork repo.

**R2.2**: Create a `sync/upstream-2026-05` branch for the merge work. Do NOT merge directly into `main`.

**R2.3**: Cherry-pick or merge the 6 identified upstream commits. Resolve conflicts preserving local RDNA2 experimental kernels, RDNA3 probes, and TurboQuant work.

**R2.4**: After merge, verify the build produces a functional binary with `llama-cli --help` and `llama-bench` passes.

**R2.5**: Run `validate_hygiene.sh` to confirm RDNA2 pipeline still works after merge.

**R2.6**: Update `DEFAULT_BRANCH` in `llama_cpp.rs` if the release branch name changes.

### R3: Documentation & Changelog

**R3.1**: Update `CHANGELOG.md` to cut the `[Unreleased]` section into a versioned `[0.2.0] - 2026-05-30` (Anagami) release, highlighting:
- All 35 installers now native Rust (zero shell script dependencies)
- Rusty Llama integration with CUDA isolation guards
- Windows Alpha support
- Upstream llama.cpp sync with BF16 flash attention + tensor parallelism

**R3.2**: Update `README.md` to:
- Feature Rusty Llama as a first-class component
- Document the CUDA isolation guarantee
- Add Windows Alpha testing call-to-action
- Update architecture diagram showing all-Rust installer path

**R3.3**: Update `docs/INSTALLER_STATUS.md` to:
- Remove the shell scripts backend diagram
- Show the Rust-native architecture
- Mark all 35 components as Rust-native

**R3.4**: Update `Fork/llama.cpp-turboquant-hip/README.md` to:
- Note the upstream sync point
- Update performance tables if new benchmarks are available
- Document the CUDA isolation enforcement by Rusty Stack

**R3.5**: Update `VERSION` file from `0.1.0` to `0.2.0`.

---

## 4. Non-Goals

- Do NOT modify the llama.cpp C++ source code for CUDA removal — the CUDA code paths are upstream and must remain for fork maintainability. Isolation is enforced at the CMake flag level.
- Do NOT add Vulkan backend support — that's a separate effort.
- Do NOT refactor installer.rs — it works, leave it.
- Do NOT change the existing pip-based installer guards — they already work.
- Do NOT create new shell scripts.

---

## 5. Validation Assertions

All new code MUST include validation assertion comments:

| Prefix | Scope |
|--------|-------|
| `VAL-GUARD-001` through `VAL-GUARD-010` | CUDA isolation guards |
| `VAL-SYNC-001` through `VAL-SYNC-005` | Upstream sync verification |
| `VAL-DOC-001` through `VAL-DOC-005` | Documentation completeness |

---

## 6. Test Matrix

| Test | Command | Expected |
|------|---------|----------|
| All Rust tests pass | `cargo test` in rusty-stack/ | 1482+ tests, 0 failures |
| New guard tests pass | `cargo test guard` | All VAL-GUARD-* tests pass |
| llama_cpp tests pass | `cargo test llama_cpp` | All llama_cpp module tests pass |
| Release build succeeds | `cargo build --release` | Clean compile |
| Fork hygiene passes | `./scripts/validate_hygiene.sh` in Fork/ | Exit 0 |

---

## 7. Risk Register

| Risk | Mitigation |
|------|-----------|
| Upstream merge conflicts with RDNA2 kernels | Cherry-pick individual commits instead of full merge |
| CUDA toolkit detection false-positives (ROCm's HIP uses CUDA API names) | Check for `nvcc` and `nvidia-smi` specifically, not CUDA env vars |
| Windows build regression from upstream sync | Test Windows build path separately |
| Binary linkage check fails on stripped binaries | `ldd` works on stripped binaries — not a risk |
