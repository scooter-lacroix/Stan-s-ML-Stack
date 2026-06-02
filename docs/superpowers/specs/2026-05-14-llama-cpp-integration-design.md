# llama.cpp-turboquant-hip Integration into Rusty Stack

**Date:** 2026-05-14
**Status:** Draft
**Reviewer:** Codex (gpt-5.4, 3 rounds, maximal scrutiny)

## Overview

Integrate the llama.cpp-turboquant-hip fork into Rusty Stack as a native component installer with COMPLETE RDNA 3 and RDNA 4 compatibility. The fork adds TurboQuant (PolarQuant + QJL KV cache compression) with new GGML types TURBO2_0/3_0/4_0 (2/3/4-bit quantization) and 15 flash attention template instances. Currently verified on RDNA 2 only.

**Target hardware:** Developer GPUs are RDNA 3 (gfx1100). All RDNA 3 HARDWARE_BLOCKER items can be empirically tested immediately. RDNA 4 items remain gated on hardware access.

---

## Section 1: Fork Hosting and Build Integration

### 1.1 Private GitHub Repo

- **Host:** `github.com/scooter-lacroix/llama.cpp-turboquant-hip`
- **Visibility:** Private (installer uses HTTPS + token or SSH)
- **Installer pulls source via `git clone`** during CMake build step

### 1.2 Build Method: CMake Source Build

- Installer clones the private repo to a temporary build directory
- Runs CMake with HIP/ROCm flags (see Section 4.3 for per-channel flags)
- Builds with `cmake --build . --target llama-cli llama-bench llama-server` (and other needed targets)
- Installs binaries to `~/.mlstack/components/llama-cpp/bin/`

### 1.3 Component Tier

- **Category:** Extension (not Core)
- **Install path:** `~/.mlstack/components/llama-cpp/`
- **Detection:** CommandBased — `llama-cli --version`

### 1.4 GPU Detection Flow

Existing Rusty Stack pattern (installer.rs:4767):
1. Run `rocminfo`
2. Scan output for `gfx*` tokens
3. Keep highest numeric value
4. Fallback to `gfx000` if detection fails

---

## Section 2: Foundation Phase

### 2.1 WARP_SIZE Decision

**CRITICAL: WARP_SIZE stays 32 for ALL RDNA generations (1/2/3/4).**

Only gfx8/gfx9 (CDNA/GCN) use wave64. The fork's code is correct as-is:
- `common.cuh:43` — `WARP_SIZE` is 32
- `common.cuh:339` — `ggml_cuda_get_physical_warp_size()` returns 32 for RDNA
- `hip.h:244` — RDNA1-4 macro definitions all map to wave32 architectures

Do NOT introduce a global WARP_SIZE=64 flip for RDNA4. If a future need arises for different logical subgroup widths, introduce a subgroup abstraction layer — but this is NOT needed now.

### 2.2 D=64 static_assert — Not a Blocker

`fattn-vec.cuh:111` has `static_assert(D % (2*WARP_SIZE) == 0)`. With WARP_SIZE=32:
- D=64: `64 % (2*32) = 64 % 64 = 0` — **PASSES**
- D=128: `128 % 64 = 0` — **PASSES**

No fix needed. This only becomes a blocker if someone flips WARP_SIZE to 64, which Section 2.1 explicitly forbids.

### 2.3 Extended Blast Radius Audit

All files using WARP_SIZE or warp_size for reduction/shuffle/shared-mem:

| File | Lines | Usage |
|------|-------|-------|
| `fattn-vec.cuh` | 65-111, 293-296, 436 | nthreads_KQ_q, shuffle reductions, shared mem KQ_max_shared |
| `fattn-common.cuh` | 150-180 | WARP_SIZE in shared mem sizing, barrier sync |
| `topk-moe.cu` | 182, 271 | WARP_SIZE/2 argmax reduction, block dims |
| `mmq.cuh` | 3486-3550, 3641, 4040-4048, 4191 | LDS double-buffer, waves_per_eu, shared mem sizing |
| `argmax.cu` | 24 | WARP_SIZE-dependent reduction |
| `set-rows.cu` | 298 | WARP_SIZE-dependent indexing |

With WARP_SIZE=32 on all RDNA, these remain correct 32-lane reductions matching hardware subgroup. No changes needed in the foundation phase.

**Open question:** `hipDeviceProp_t::warpSize` runtime return value on gfx1200 (RDNA4). If this returns 64 despite hardware doing wave32, host/device mismatch exists at `ggml-cuda.cu:248`. Must test empirically on real RDNA4 hardware.

### 2.4 Reconcile with Existing RDNA3/4 Code

The fork already has RDNA3/4 support code that must NOT be broken:

1. **RDNA3/RDNA4 macros** — `hip.h:244`
2. **MMQ path selection** — `mmq.cu:342-343`: RDNA3 falls into WMMA branch (but inner per-format ops still use DP4A)
3. **MMVQ heuristics** — `mmvq.cu:327` (RDNA3), `mmvq.cu:351` (RDNA4)
4. **WMMA flash attention** — `fattn-wmma-f16.cuh` gated by `GGML_HIP_ROCWMMA_FATTN`
5. **DP4A dispatch** — `common.cuh:668`: sudot4 for RDNA3/4

**UNVERIFIED:** WMMA flash attention runtime correctness on RDNA3 hardware. The file `fattn-wmma-f16.cu:1` is marked "old and deprecated". Must validate before building on it (see Section 3.1). **Developer has RDNA3 hardware — can test empirically.**

### 2.5 Detection: CommandBased + Runtime Verification

Replace initial PathBased proposal with multi-layer verification:

1. **Registry detection:** CommandBased — run `llama-cli --version`, check exit code, parse version
2. **Verification (component_status.rs):** `llama-cli --help` (functional) + `ldd $(which llama-cli) | grep rocblas` (linked correctly) + `llama-bench -p 1 -n 1` with tiny model (smoke test)
3. **Enhanced verification:** Full benchmark run via `llama-bench`

### 2.6 ROCm Channel Feature Gating

**Channel definitions (rocm.rs:21 — verified):**
- Legacy: ROCm 6.4.3
- Stable: ROCm 7.1
- Latest: ROCm 7.2.2

**rocWMMA versions:**
- Legacy (6.4.3): rocWMMA 1.7.0 — excluded by `ROCWMMA_VERSION_MAJOR > 1` check in `fattn-wmma-f16.cuh:14`
- Stable (7.1): rocWMMA 2.0.0
- Latest (7.2.2): rocWMMA 2.2.0

**Feature matrix:**

| Channel | ROCm | RDNA2 | RDNA3 | RDNA4 | WMMA FA | rocWMMA |
|---------|------|-------|-------|-------|---------|---------|
| Legacy | 6.4.3 | Yes | No | No | No | 1.7 (excluded) |
| Stable | 7.1 | Yes | Yes | No | Yes | 2.0 |
| Latest | 7.2.2 | Yes | Yes | Yes | Yes | 2.2 |

---

## Section 3: RDNA3/4 Aggressive Optimization Scope

### Phase 2: RDNA3 Enablement

#### 3.1 WMMA Flash Attention Validation (prerequisite)

The existing WMMA FA path at `fattn-wmma-f16.cu` is marked "old and deprecated" (line 1). Before building on it:

- Validate it produces correct results on RDNA3 hardware (gfx1100/gfx1101)
- Test with rocWMMA 2.0.0 (ROCm 7.1) and 2.2.0 (ROCm 7.2.2)
- If deprecated path is broken, evaluate upstream llama.cpp's current WMMA FA implementation for merge
- Determine if `GGML_HIP_ROCWMMA_FATTN` should be replaced with a newer implementation

**Files:** `fattn-wmma-f16.cu:1`, `fattn-wmma-f16.cuh:9-15`, `fattn.cu:539` (dispatch)

#### 3.2 nthreads_KQ_q Tuning for RDNA3

`fattn-vec.cuh:65-75` sets `nthreads_KQ_q=2` for ALL RDNA. RDNA3 has more VGPRs and WMMA:

- RDNA3: restore `nthreads_KQ_q=4` (matching NVIDIA/cuda path)
- RDNA2: keep `nthreads_KQ_q=2` (conservative, proven)
- Gate via runtime CC check: if RDNA3 CC range, use 4; else 2

**Template instances:** 134 generated files in `template-instances/`. Instances are parameterized by D/type_K/type_V — NOT by nthreads. Zero new template files needed for this change.

**Files:** `fattn-vec.cuh:65-75`

#### 3.3 RDNA3 LDS Opt-in (128KB capability)

RDNA3 CUs have up to 128KB LDS (vs 64KB on RDNA2). Current issues:

- `CUDA_SET_SHARED_MEMORY_LIMIT` is a no-op on HIP (`common.cuh:215-218`)
- `smpbo` uses non-optin value (`ggml-cuda.cu:259`)

For RDNA3:
- Implement HIP shared memory opt-in using `hipFuncSetAttribute` with `hipFuncAttributeMaxDynamicSharedMemoryBytes`
- Fix smpbo to use `sharedMemPerBlockOptin` on HIP where available
- This enables larger MMQ tile sizes on RDNA3, improving matmul throughput

**CAVEAT:** `hipFuncSetAttribute` EXISTS in HIP but ROCm headers say it "is ignored" on AMD devices. The fork already uses it in `fattn-mma-f16.cuh:1781`. Must test empirically whether it actually enables >64KB shared memory on RDNA3.

**Files:** `common.cuh:205-219`, `ggml-cuda.cu:259`, `mmq.cuh:4040-4048`, `mmq.cuh:4191`

#### 3.4 RDNA3 Occupancy Tuning

The `amdgpu_waves_per_eu(4,8)` at `mmq.cuh:3641` was tuned for RDNA2 wave32. For RDNA3:

- Evaluate if wave32 mode with 4-8 waves is still optimal
- Consider RDNA3's dual compute engine per CU — occupancy characteristics differ
- Tune per kernel-class: MMQ, flash attention, MoE top-k, dequant

**Note:** Only 1 `amdgpu_waves_per_eu` annotation in the entire fork (`mmq.cuh:3641`). But 30+ `__launch_bounds__` annotations across multiple files.

**Files:** `mmq.cuh:3637-3642`

#### 3.5 RDNA3 MoE Double-Buffer Adaptation

The RDNA2_MATMUL_OPT_V1 double-buffering in `mmq.cuh:3486-3550` is gated to RDNA2 only. For RDNA3:

- Adapt the double-buffer pattern for RDNA3's larger LDS (128KB enables bigger tiles)
- Adjust `lds_bank_pad` (bank count is same 32, so padding stays valid)
- New compile flag: `RDNA3_MATMUL_OPT_V1` with runtime gate checking `GGML_CUDA_CC_IS_RDNA3`
- Evaluate if WMMA path makes double-buffering unnecessary (WMMA may already be fast enough)

**Files:** `mmq.cuh:3486-3550`, `mmq.cuh:4059-4068`, `mmq.cu:342`

### Phase 3: RDNA4 Enablement

#### 3.6 RDNA4 Wave Mode Confirmation

**CRITICAL UNVERIFIED:** Does `hipDeviceProp_t::warpSize` return 32 or 64 on RDNA4 (gfx1200)?

- If warpSize=32: RDNA4 works identically to RDNA3, all existing code is correct, no wave64 handling needed
- If warpSize=64: Major surgery needed — all WARP_SIZE-dependent code requires wave64-aware variants

Must be determined empirically before any RDNA4 work begins. Options:
1. Test on real RDNA4 hardware
2. Check AMD ROCm documentation for `hipDeviceProp_t::warpSize` on gfx12xx
3. Check ROCm source code for the warpSize initialization per arch

#### 3.7 RDNA4 WMMA Support

RDNA4 (gfx1200) introduces next-gen WMMA. The fork's WMMA availability check needs RDNA4 support:

- `common.cuh:313` (`amd_wmma_available`) — already has RDNA4 range defined at `common.cuh:313-314`
- Verify rocWMMA 2.2.0 (ROCm 7.2.2) supports gfx1200 WMMA operations
- Check if RDNA4 WMMA has different fragment dimensions than RDNA3

**Files:** `common.cuh:290-320`, `common.cuh:60-104`

#### 3.8 RDNA4 Flash Attention

- If RDNA4 warpSize=32: flash attention works as-is with the vec kernel path (`fattn-vec`)
- If RDNA4 warpSize=64: need wave64-aware flash attention variant with modified reduction widths
- The WMMA path (Section 3.1) may be the right answer for RDNA4 regardless — WMMA fragment dimensions are independent of wave size

**Files:** `fattn-vec.cuh:65-111`, `fattn-wmma-f16.cuh`, `fattn.cu:539`

---

## Section 4: Rusty Stack Integration Plan

### 4.1 Files to Modify (12+ files)

Traced from flash_attention_ck integration pattern:

| # | File | Change |
|---|------|--------|
| 1 | `src/installers/components/llama_cpp.rs` | **NEW:** Component installer module — CMake source build from private repo |
| 2 | `src/installers/components/mod.rs` | Add `pub mod llama_cpp;`, re-export, add to NATIVE_COMPONENT_IDS, add dependency on "rocm" |
| 3 | `src/installer.rs` | Add match arm for "llama-cpp" in `run_native_installer()` |
| 4 | `src/platform/registry.rs` | Add ComponentInfo entry — CommandBased detection via `llama-cli --version` |
| 5 | `src/state.rs` | Add Component entry in `default_components()` with Extension category |
| 6 | `src/component_status.rs` | Add installed detection + verification hooks (llama-bench smoke test, ldd rocblas check) |
| 7 | `src/verification/mod.rs` | Add to `enhanced_verify()` component list |
| 8 | `src/orchestrator/planner.rs` | Add to update planner if component needs update tracking |
| 9 | `src/adapter/mod.rs` | Add adapter entry for the component |
| 10 | `src/adapter/legacy_adapter.rs` | Add legacy fallback if needed |
| 11 | `src/platform/windows.rs` | Add stub or unsupported entry |
| 12 | `src/core/fixtures/baseline_manifest.json` | Add component entry for test fixtures |

### 4.2 Detection Strategy

**Registry detection (registry.rs):** CommandBased
- Run `llama-cli --version`
- Check exit code
- Parse version string

**Component verification (component_status.rs):**
- `llama-cli --help` — functional check
- `ldd $(which llama-cli) | grep rocblas` — verify ROCm linkage
- `llama-bench -p 1 -n 1` — smoke test with tiny model (no existing ldd-style linkage check pattern in codebase — this is new)

**Enhanced verification:** Full benchmark run via `llama-bench`

### 4.3 Installer CMake Flags Per Channel

CMake flag format: `GPU_TARGETS=` is primary flag (docs/build.md:360). `AMDGPU_TARGETS` works via forwarding chain: AMDGPU_TARGETS -> GPU_TARGETS -> CMAKE_HIP_ARCHITECTURES.

```
Legacy:  -DGGML_HIP=ON -DGPU_TARGETS=gfx1030
Stable:  -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DGPU_TARGETS=gfx1030;gfx1100;gfx1101
Latest:  -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DGPU_TARGETS=gfx1030;gfx1100;gfx1101;gfx1200
```

### 4.4 Installer Architecture (llama_cpp.rs)

Based on flash_attention_ck.rs (line 45) and onnxruntime.rs (line 23) patterns:

```rust
// HipArchs enum — per-family matching onnxruntime.rs:25 pattern
enum HipArchs {
    Gfx1030,                          // RDNA2 (gfx1030-gfx1037)
    Gfx1100,                          // RDNA3 (gfx1100-gfx1101)
    Gfx1200,                          // RDNA4 (gfx1200-gfx1201)
    Default,                          // Fallback
}

impl HipArchs {
    fn gpu_targets(&self, channel: &str) -> String {
        match (self, channel) {
            (Gfx1030, _) => "gfx1030".into(),
            (Gfx1100, "legacy") => "gfx1030".into(), // fallback to RDNA2
            (Gfx1100, _) => "gfx1030;gfx1100;gfx1101".into(),
            (Gfx1200, "legacy") => "gfx1030".into(),
            (Gfx1200, "stable") => "gfx1030;gfx1100;gfx1101".into(),
            (Gfx1200, _) => "gfx1030;gfx1100;gfx1101;gfx1200".into(),
            _ => "gfx1030".into(), // safe default
        }
    }
}

// CMake command construction
fn build_cmake_command(channel, gpu_arch) -> Command {
    // 1. git clone private repo to temp build dir
    // 2. cmake -B build with per-channel flags
    // 3. cmake --build build --target llama-cli llama-bench llama-server
    // 4. cmake --install build --prefix ~/.mlstack/components/llama-cpp
}
```

### 4.5 ROCm Channel Gating Logic

```rust
fn cmake_flags_for_channel(channel: &str, gpu_arch: &str) -> Vec<String> {
    let mut flags = vec!["-DGGML_HIP=ON".into()];

    match channel {
        "legacy" => {
            // ROCm 6.4.3 — RDNA2 only, no WMMA FA
            flags.push(format!("-DGPU_TARGETS=gfx1030"));
        }
        "stable" => {
            // ROCm 7.1 — RDNA2+RDNA3, WMMA FA enabled
            flags.push("-DGGML_HIP_ROCWMMA_FATTN=ON".into());
            flags.push(format!("-DGPU_TARGETS=gfx1030;gfx1100;gfx1101"));
        }
        "latest" => {
            // ROCm 7.2.2 — RDNA2+RDNA3+RDNA4, full WMMA
            flags.push("-DGGML_HIP_ROCWMMA_FATTN=ON".into());
            flags.push(format!("-DGPU_TARGETS=gfx1030;gfx1100;gfx1101;gfx1200"));
        }
    }
    flags
}
```

---

## Open Questions — Resolved and Remaining

### Q1: RDNA4 warpSize — RESOLVED: 32
`hipDeviceProp_t::warpSize` returns **32** on gfx1200/gfx1201. AMD HIP docs state "AMD devices return 64 for gfx9 and 32 for gfx10 and above." rocWMMA docs list gfx1200/gfx1201 under "RDNA architectures (wave32)." No host/device mismatch exists.

**Impact:** No wave64 handling needed for RDNA4. All existing WARP_SIZE=32 code is correct.

Sources: HIP C++ language extensions (rocmdocs.amd.com), rocWMMA API reference (rocm.docs.amd.com)

### Q2: RDNA4 LDS size — RESOLVED: 64KB per CU
AMD's RDNA4 ISA guide lists LDS as "64kB." No AMD/ROCm source confirms 128KB opt-in shared memory on gfx1200/gfx1201. The LDS opt-in work (Section 3.3) is RDNA3-specific.

**Impact:** RDNA4 does NOT benefit from LDS opt-in. smpbo fix still needed for RDNA3.

**Remaining verification:** Query `sharedMemPerBlock`, `sharedMemPerBlockOptin`, `sharedMemPerMultiprocessor` on real gfx1200 hardware to confirm HIP reporting.

Sources: RDNA4 ISA guide (docs.amd.com), HIP hardware implementation docs (rocm.docs.amd.com)

### Q3: hipFuncSetAttribute effectiveness — UNVERIFIED (testable on developer RDNA3 hardware)
The API exists in ROCm but docs say some hints are "ignored on AMD devices." The fork already calls it at fattn-mma-f16.cuh:1781. No authoritative source confirms it actually enables >64KB dynamic shared memory on RDNA3.

**Needed test:** On real gfx1100, print `sharedMemPerBlockOptin`, call `hipFuncSetAttribute(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, N)`, then launch kernels with 64KB/96KB/128KB dynamic shared memory. Developer has RDNA3 — can run this immediately.

**Impact on implementation:** The smpbo bug fix (ggml-cuda.cu:259: use `sharedMemPerBlockOptin` instead of `sharedMemPerBlock`) is worth doing regardless — it's a bug that MUSA/NVIDIA paths already handle correctly. But whether it enables >64KB usage depends on this empirical test.

### Q4: WMMA FA correctness on RDNA3 — UNVERIFIED (testable on developer RDNA3 hardware)
Upstream llama.cpp issue #13110 (2025-04-25) confirms the WMMA FA path was still actively built for gfx1201. No authoritative test report proves numerical correctness on gfx1100. No accuracy bug report found either.

**Needed test:** Run FA-vs-non-FA output comparison on real gfx1100 across multiple head sizes / KV types / context lengths with max-abs/max-rel error thresholds. Developer has RDNA3 — can run this immediately.

**Impact on implementation:** Phase 2 MUST validate WMMA FA on RDNA3 before building optimization layers on top of it. If broken, fallback to vec kernel path.

---

## Risks and Mitigations

| Risk | Mitigation | Status |
|------|------------|--------|
| Private repo access from installer | Support both SSH key and HTTPS token auth | Open |
| WMMA FA deprecated/broken on RDNA3 | Evaluate upstream llama.cpp WMMA FA for merge; fallback to vec kernel | UNVERIFIED — testable on developer RDNA3 hardware |
| RDNA4 wave64 mode | ~~Empirical test required~~ | RESOLVED — warpSize=32 confirmed |
| hipFuncSetAttribute no-op on AMD | Test empirically; if truly no-op, skip LDS opt-in and work within 64KB limit | UNVERIFIED — testable on developer RDNA3 hardware |
| RDNA4 LDS >64KB | ~~Check if 128KB like RDNA3~~ | RESOLVED — 64KB confirmed |
| Template instance explosion | Confirmed: nthreads_KQ_q change needs ZERO new template files | RESOLVED |
| CMake flag format | Verified: GPU_TARGETS is primary | RESOLVED |
| smpbo bug on HIP | Fix ggml-cuda.cu:259 to use sharedMemPerBlockOptin | Confirmed bug |

---

## Implementation Phases

### Phase 1: Foundation + Installer (no kernel changes)

#### P1.1 — Create private GitHub repo
- Create `scooter-lacroix/llama.cpp-turboquant-hip` as private repo
- Push current fork content
- **Auth mechanism needed:** Current source-build installers use plain `git clone <url>` (installer.rs:2315, fastvideo.rs:17). No credential injection surface exists. Options:
  - SSH deploy key on build machine (pre-configured)
  - HTTPS PAT via environment variable (e.g., `LLAMA_CPP_REPO_TOKEN`)
  - Bundle source tarball instead of git clone (avoids auth entirely)
- Decision needed before implementation

#### P1.2 — Create llama_cpp.rs installer module
File: `src/installers/components/llama_cpp.rs` (NEW)
- CMake source build from private repo (git clone → cmake → make → install)
- **HipArchs enum pattern:** Use per-family variants matching onnxruntime.rs:25 (`Gfx1030`/`Gfx1100`/`Gfx1200`/`Default`), NOT grouped supersets
- Map each variant to semicolon-separated GPU_TARGETS string at build time
- ROCm channel gating: Legacy=RDNA2 only, Stable=RDNA2+3, Latest=RDNA2+3+4
- CMake flags: `-DGGML_HIP=ON`, `-DGGML_HIP_ROCWMMA_FATTN=ON` (Stable/Latest only), `-DGPU_TARGETS=...`
- **Build targets:** `llama-cli`, `llama-bench`, `llama-server` (tools/cli/CMakeLists.txt, tools/llama-bench/CMakeLists.txt, tools/server/CMakeLists.txt)
- **Install:** `cmake --install` with CMAKE_INSTALL_PREFIX
- Dependency on "rocm" component
- **Note:** `llama-cli` does NOT support `--version` flag (verified: no `--version`/`print_version` in tools/cli source). Detection must use `--help` exit code or different strategy.

#### P1.3 — Wire into mod.rs
File: `src/installers/components/mod.rs`
- Add `pub mod llama_cpp;` (module declaration ~line 68)
- Add re-export (~line 93)
- Add to NATIVE_COMPONENT_IDS (~line 135)
- Add dependency on "rocm" (~line 199)
- Update hardcoded component count assertions if present (~line 127, 194, 289)

#### P1.4 — Wire into installer.rs
File: `src/installer.rs`
- Add match arm for "llama-cpp" in run_native_installer()
- Call llama_cpp installer

#### P1.5 — Add registry detection
File: `src/platform/registry.rs`
- Add ComponentInfo entry in known_components() (~line 68-125)
- **CRITICAL (Codex finding #2):** CommandBased detection alone is insufficient because `llama-cli` is not on PATH by default (installed to `~/.mlstack/components/llama-cpp/bin/`). Options:
  - **Option A:** Use PathBased detection pointing to `~/.mlstack/components/llama-cpp/bin/llama-cli`
  - **Option B:** Add PATH export for component binaries (see P1.6), then use CommandBased
  - **Option C:** Hybrid — PathBased for primary, CommandBased as fallback
- Update hardcoded allowlists (~line 809, 821, 1276)

#### P1.6 — PATH/export for component binaries (NEW — Codex finding #1)
File: `src/installers/components/permanent_env.rs`, `src/platform/environment.rs`
- Current env generation only exports ROCm bins (permanent_env.rs:144, environment.rs:354)
- Must add export for `~/.mlstack/components/llama-cpp/bin/` so detection finds `llama-cli`
- OR: use PathBased detection with known install prefix

#### P1.7 — Add state entry
File: `src/state.rs`
- Add Component entry in default_components() (~line 251-460) with Extension category

#### P1.8 — Add component status verification
File: `src/component_status.rs`
- Add to is_component_installed_by_id() match (~line 143)
- Add verification commands:
  - `llama-cli --help` — functional check (exit code 0)
  - `llama-bench --help` or `llama-bench --list-devices` — **NOT** `llama-bench -p 1 -n 1` (Codex finding #5: requires model file, defaults to `models/7B/...`)
  - Linkage check: `ldd` against `amdhip64` or `hipblas` — **NOT** `rocblas` (Codex finding #10: fork links against hipblas/hip, not rocblas directly)
- Add to enhanced_verification_commands() (~line 549)
- Add to helper functions (~line 649)

#### P1.9 — Wire verification module
File: `src/verification/mod.rs`
- Add to enhanced_verify_components() list (~line 55)

#### P1.10 — Wire orchestrator planner
File: `src/orchestrator/planner.rs`
- Add to ROCm requirements allowlist (~line 523)
- Add dependency derivation (~line 631)

#### P1.11 — Wire adapter
File: `src/adapter/mod.rs`
- Add adapter entry (~line 559) — mostly registry-driven, minimal hand-edit needed

#### P1.12 — Wire legacy adapter
File: `src/adapter/legacy_adapter.rs`
- Add legacy fallback (~line 180) if component supports legacy install path

#### P1.13 — Windows stub
File: `src/platform/windows.rs`
- Add stub or unsupported entry (~lines 219, 243, 300)

#### P1.14 — Test fixtures
File: `src/core/fixtures/baseline_manifest.json`
- Add component entry (~line 49)

#### P1.15 — Build and test
- `cargo build --release` in rusty-stack/
- Test installer flow
- Test detection, installation, and verification

**Parallelization (P1):**
- P1.3, P1.4, P1.7, P1.9, P1.13, P1.14 can run in parallel (distinct integration surfaces)
- P1.5, P1.6, P1.8 must stay grouped (detection/verification contract)
- P1.1 must complete before P1.2
- P1.2 must complete before P1.4

**HARDWARE_BLOCKER:** P1.15 requires real RDNA2 GPU or equivalent CI

### Phase 2: RDNA3 Enablement (kernel changes in fork)

#### P2.1 — Fix smpbo bug
File: `Fork/llama.cpp-turboquant-hip/ggml/src/ggml-cuda/ggml-cuda.cu:259`
- Change `info.devices[id].smpbo = prop.sharedMemPerBlock;` to `info.devices[id].smpbo = prop.sharedMemPerBlockOptin;`
- CONFIRMED BUG: MUSA (line 279) and NVIDIA (line 286) already use Optin
- Impact: correct shared memory reporting on RDNA3 (may report 128KB vs 64KB)
- **No hardware needed** — this is a correctness fix

#### P2.2 — Fix CUDA_SET_SHARED_MEMORY_LIMIT no-op on HIP
File: `Fork/llama.cpp-turboquant-hip/ggml/src/ggml-cuda/common.cuh:215-218`
- Currently a no-op on HIP: `#if !defined(GGML_USE_HIP)` guards the real implementation
- For RDNA3: implement the shared memory limit call using `hipFuncSetAttribute`
- Depends on P2.4 for empirical verification that hipFuncSetAttribute actually works
- **Also touches:** mmq.cuh:4074 (calls CUDA_SET_SHARED_MEMORY_LIMIT)

#### P2.3 — Validate WMMA FA on RDNA3 **[TESTABLE — developer has RDNA3 gfx1100]**
- Build fork with `GGML_HIP_ROCWMMA_FATTN=ON` for gfx1100
- Run correctness tests: FA-vs-non-FA output comparison on real gfx1100
- **Dispatch chain:** fattn-wmma-f16.cuh:9 (compile gate) → fattn.cu:539 (runtime dispatch)
- If broken: evaluate upstream llama.cpp WMMA FA for merge, or fallback to vec kernel
- If working: proceed with WMMA optimization

#### P2.4 — Tune nthreads_KQ_q for RDNA3 **[REVISED — Codex finding #3]**
File: `Fork/llama.cpp-turboquant-hip/ggml/src/ggml-cuda/fattn-vec.cuh:65-75`
- **CORRECTION:** `nthreads_KQ_q` is a compile-time `constexpr` used as a template argument (fattn-vec.cuh:67→81→90). Cannot be switched at runtime.
- **Revised approach:** Compile-time split — build separate template instantiations for RDNA3 with `nthreads_KQ_q=4` and use runtime dispatch in fattn.cu to select the right kernel based on CC
- This MAY require additional template instances despite earlier claim of "zero new files"
- Must audit: fattn.cu dispatch logic and template-instances/ generator

#### P2.5 — hipFuncSetAttribute empirical test **[TESTABLE — developer has RDNA3 gfx1100]**
- On real gfx1100: query `sharedMemPerBlockOptin`
- Test `hipFuncSetAttribute` with 96KB/128KB
- Launch kernels with >64KB dynamic shared memory
- If works: proceed with P2.6
- If doesn't work: skip LDS opt-in, document limitation

#### P2.6 — RDNA3 LDS opt-in **[depends on P2.1, P2.2, P2.5]**
- If hipFuncSetAttribute works:
  - Implement shared memory opt-in for MMQ on RDNA3
  - Larger tile sizes via 128KB LDS
  - Gate: compile flag + runtime RDNA3 CC check
- **Note:** Tile selection is bounded by smpbo (mmq.cuh:4178), so P2.1 is prerequisite

#### P2.7 — RDNA3 occupancy tuning **[TESTABLE — developer has RDNA3 gfx1100]**
File: `Fork/llama.cpp-turboquant-hip/ggml/src/ggml-cuda/mmq.cuh:3637-3642`
- Evaluate `amdgpu_waves_per_eu(4,8)` for RDNA3
- Per-kernel-class tuning: MMQ, FA, MoE, dequant
- Requires real hardware for benchmark validation — developer has gfx1100

#### P2.8 — RDNA3 MoE double-buffer adaptation **[TESTABLE — developer has RDNA3 gfx1100]**
File: `Fork/llama.cpp-turboquant-hip/ggml/src/ggml-cuda/mmq.cuh:3486-3550`
- Adapt RDNA2 double-buffer pattern for RDNA3
- New compile flag: `RDNA3_MATMUL_OPT_V1`
- Runtime gate: `GGML_CUDA_CC_IS_RDNA3`
- Should wait until P2.3/P2.5 decide viable kernel paths

**Parallelization (P2):**
- P2.1 is standalone (no hardware, do first)
- P2.3 and P2.5 can run in parallel — developer has RDNA3 hardware
- P2.2, P2.6 depend on P2.5 results
- P2.4 is standalone code work but may need hardware validation
- P2.7, P2.8 should wait until P2.3/P2.5 resolve

### Phase 3: RDNA4 Enablement (kernel changes in fork)

#### P3.1 — Build test on RDNA4 **[HARDWARE_BLOCKER]**
- Build fork with `GPU_TARGETS=gfx1200`
- Verify compilation succeeds
- Test basic functionality on real gfx1200 hardware

#### P3.2 — WMMA support for RDNA4 **[HARDWARE_BLOCKER]**
File: `Fork/llama.cpp-turboquant-hip/ggml/src/ggml-cuda/common.cuh:313`
- Already has RDNA4 range in `amd_wmma_available()`
- Verify rocWMMA 2.2.0 supports gfx1200 WMMA operations
- Check if RDNA4 WMMA has different fragment dimensions

#### P3.3 — Flash attention on RDNA4 **[HARDWARE_BLOCKER]**
- warpSize=32 confirmed — vec kernel path works as-is
- WMMA path available via fattn-mma-f16.cuh:527 gate (`AMD_WMMA_AVAILABLE && RDNA4`)
- **Dispatch chain:** fattn-wmma-f16.cuh:18 (compile gate) → fattn.cu:539 (runtime dispatch)
- Validate both paths on real hardware

**Parallelization (P3):**
- P3.1 can start independently (just compilation)
- P3.2, P3.3 need P3.1 and real hardware
