# llama.cpp-turboquant-hip Integration into Rusty Stack

**Date:** 2026-05-14
**Status:** Draft
**Reviewer:** Codex (gpt-5.4, 3 rounds, maximal scrutiny)

## Overview

Integrate the llama.cpp-turboquant-hip fork into Rusty Stack as a native component installer with COMPLETE RDNA 3 and RDNA 4 compatibility. The fork adds TurboQuant (PolarQuant + QJL KV cache compression) with new GGML types TURBO2_0/3_0/4_0 (2/3/4-bit quantization) and 15 flash attention template instances. Currently verified on RDNA 2 only.

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

**UNVERIFIED:** WMMA flash attention runtime correctness on RDNA3 hardware. The file `fattn-wmma-f16.cu:1` is marked "old and deprecated". Must validate before building on it (see Section 3.1).

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
// HipArchs enum — per-channel GPU targeting
enum HipArchs {
    Gfx1030,                          // Legacy: RDNA2 only
    Gfx1030_1100_1101,                // Stable: RDNA2 + RDNA3
    Gfx1030_1100_1101_1200,           // Latest: RDNA2 + RDNA3 + RDNA4
    Default,                          // Fallback
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

## Open Questions (Require Empirical Testing)

1. **RDNA4 warpSize:** What does `hipDeviceProp_t::warpSize` return on gfx1200? (32 or 64)
2. **RDNA4 LDS size:** 64KB (like RDNA2) or 128KB (like RDNA3)?
3. **WMMA FA correctness:** Does the deprecated `fattn-wmma-f16.cu` produce correct results on RDNA3?
4. **hipFuncSetAttribute effectiveness:** Does `hipFuncSetAttribute` actually enable >64KB shared memory on RDNA3 despite ROCm docs saying "ignored"?

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Private repo access from installer | Support both SSH key and HTTPS token auth |
| WMMA FA deprecated/broken on RDNA3 | Evaluate upstream llama.cpp WMMA FA for merge; fallback to vec kernel |
| RDNA4 wave64 mode | Empirical test required before any RDNA4 work; if wave64, scope increases significantly |
| hipFuncSetAttribute no-op on AMD | Test empirically; if truly no-op, skip LDS opt-in and work within 64KB limit |
| Template instance explosion | Confirmed: nthreads_KQ_q change needs ZERO new template files (instances parameterized by D/type only) |
| CMake flag format | Verified: GPU_TARGETS is primary; AMDGPU_TARGETS works via forwarding chain |

---

## Implementation Phases

### Phase 1: Foundation + Installer (no kernel changes)
- Create private GitHub repo
- Build Rust installer component (12+ files)
- CMake source build with per-channel flags
- CommandBased detection + verification
- Test on RDNA2 (already works)

### Phase 2: RDNA3 Enablement
- Validate WMMA FA on RDNA3 (Section 3.1)
- Tune nthreads_KQ_q for RDNA3 (Section 3.2)
- LDS opt-in if hipFuncSetAttribute works (Section 3.3)
- Occupancy tuning (Section 3.4)
- MoE double-buffer adaptation (Section 3.5)

### Phase 3: RDNA4 Enablement
- Confirm wave mode empirically (Section 3.6)
- WMMA support (Section 3.7)
- Flash attention path (Section 3.8)
