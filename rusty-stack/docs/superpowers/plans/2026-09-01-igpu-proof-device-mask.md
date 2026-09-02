<!-- Parent: ../../../../docs/rolling-changelog.md -->
<!-- Generated: 2026-09-01 -->

# iGPU-Proof Device Mask — Investigation, Analysis & Plan

**Trigger:** Operator re-enables the 7800X3D Raphael iGPU (desktop usage while the
7900 XTX is pinned to vfio for guest passthrough). The iGPU must never leak into
the ML env, and the mask must survive PCIe enumeration changes (iGPU agent
insertion, dGPU removal, re-ordering) without regenerating garbage.

## 1. Ground truth — probed on THIS box (ROCm 7.2.4, not assumed)

Current topology: Agent 1 = CPU, Agent 2 = RX 7900 XTX (`GPU-36ebd038d4f13ef8`,
03:00.0), Agent 3 = RX 7800 XT (`GPU-7e39f0afe2a07bc1`, 16:00.0). No iGPU (disabled).

| Probe | Result |
|---|---|
| `ROCR_VISIBLE_DEVICES=0,1` | 2 GPUs ✓ (today's env works) |
| `ROCR_VISIBLE_DEVICES=2,3` | **0 GPUs** — raw ROCr *agent* indices select nothing |
| `ROCR_VISIBLE_DEVICES=1,2` | 1 GPU (7800 XT) — **compacted GPU ordinals**, iGPU/CPU agents not counted |
| `ROCR_VISIBLE_DEVICES=GPU-<uid>` (rocminfo Uuid) | exactly that 1 GPU ✓ |
| `HIP_VISIBLE_DEVICES=GPU-<uid>` via managed torch | count 1, right name ✓; both uuids → 2, right names+order ✓; ROCR+HIP+CUDA all-uuid → 2 ✓ |
| `ROCR_VISIBLE_DEVICES=03:00.0` / `0000:03:00.0` | **PCI-slot tokens unsupported** |
| Stale/absent uuid (`GPU-deadbeef…`) in ROCR / HIP | 0 GPUs, clean fail-closed (no iGPU leak) |
| sysfs `/sys/class/drm/cardN/device/unique_id` | present on both cards, matches rocminfo Uuid |
| Ambient `CUDA_VISIBLE_DEVICES=0,1` conflict | HIP supersedes CUDA (rocr agent.cpp warning) |

**Consequences:**
- Ordinal masks are enumeration-fragile by construction (iGPU insertion shifts
  every dGPU ordinal; vfio removal shifts again).
- UUID masks are enumeration-immune and degrade fail-closed — the desired
  behavior when a dGPU is pinned away.
- The installed env's mask (`0,1`) is only correct *by luck of compaction* today.

## 2. Failure matrix (current code, `feat/freetoken-rocm`)

| Scenario | env_setup fallbacks | installer merge path (rocm_index) | outcome |
|---|---|---|---|
| iGPU enabled (APU agent before dGPUs) | compacted `0,1` → **iGPU exposed as device 0** | agent-index `2,3` (or shifted) → **0 usable GPUs** (or iGPU leak) | broken |
| XTX pinned to vfio (UUID still valid, device absent) | — | mask `1,2`/`0,1` stale → ordinal 1 now = 7800 XT? wrong-device or empty | broken |
| XTX pinned + iGPU on | compaction renumbers again | stale agent ordinals | broken |
| rocminfo absent | lspci/render compacted `0,1` | compacted `0,1` | works only iGPU-less |
| heterogeneous dGPU archs (gfx1100+gfx1101) | n/a | n/a | mask fine; fastvideo mask consumer is ordinal-only |

Root causes:
1. **Three divergent detectors** — `env_setup.rs::detect_discrete_gpus`
   (compacted indices), `gpu.rs::detect_discrete_amd_gpus` (agent indices via
   rocminfo name-matching), `installer.rs` mask assembly. `gpu.rs` itself is the
   good classifier; the mask *semantics* are the bug.
2. **`rocm_index` is the wrong ordinal space** — empirically raw agent indices
   select nothing on ROCm 7.2.4 (`ROCR=2,3` → 0 GPUs). The gpu.rs doc comment
   claiming "authoritative agent index for ROCR_VISIBLE_DEVICES" is wrong.
3. **No stable device identity in the mask** — ordinals describe a *position in
   today's enumeration*, not a *device*.

## 3. Design — `runtime_device_mask()`

One builder, used by every emission path (merge + regenerate + fish mirror):

```
pub enum DeviceToken { Uuid(String), Ordinal(u32) }   // Display → "GPU-<uid>" | "N"
pub fn runtime_device_mask(dgpus: &[AmdGpu]) -> String
```

- **UUID mode (primary):** when *every* dGPU carries a uuid (`sysfs unique_id`,
  rocminfo `Uuid` fallback), emit `GPU-<uid1>,GPU-<uid2>,…` in `pci_slot` order
  (matches `detect_discrete_amd_gpus` sort → `PYTORCH_ROCM_DEVICE=0` stays the
  pci-order primary). Enumerated envs (torch, vLLM, MIGraphX) see the dGPUs in
  mask order regardless of PCIe layout. All RDNA2+ expose `unique_id`; matrix-safe.
- **Ordinal mode (fallback):** otherwise — only if every dGPU has a runtime
  ordinal — emit **runtime ordinals**: the compact 0-based position of each dGPU
  among GPU-type agents as the *runtime* enumerates them (CPU agents not
  numbered, iGPU GPU-agent counted). Derived by matching dgpus against a fresh
  rocminfo agent scan in order.
- **Neither → `Err` (no compacted tier).** A card with neither identity
  (pre-RDNA2 without rocminfo) fails LOUD instead of silently emitting compacted
  `0..n` — probed on ROCm 7.2.4, compacted ordinals ARE the iGPU-unsafe token
  space, and a silent downgrade would reintroduce the bug at the exact moment a
  dGPU is pinned. Masks are UNIFORM, never mixed: `ROCR=GPU-<uuid>,1` resolves
  both devices, but HIP/torch resolve ONLY the UUID token (layer skew).
- **Stale after topology change** (e.g. pinned dGPU): runtime filters resolve
  fewer/zero devices → ML workloads fail loud/clean; **iGPU can never appear**
  (its UUID is not in the mask; ordinals are re-derived at next refresh, not baked).
- **Refresh surface:** any `rusty-stack` run that already writes the env
  (install/repair/verify-env paths) re-derives the mask from live topology. No
  new daemon, no cron.

`GPU_ARCH`/`HSA_OVERRIDE` stay primary-dGPU-derived (unchanged); `PYTORCH_ROCM_DEVICE`
stays `0` (post-filter visible set).

## 4. Consumer compatibility (whole crate, not just this box)

| Consumer | Today | Change |
|---|---|---|
| torch/vLLM/MIGraphX/RCCL (runtime consumers) | consume mask verbatim | none (probed) |
| `rccl.rs` probe | token pass-through (`parse_visibility_tokens`) | none; add a UUID round-trip test |
| `benchmarks/mod.rs` `_gpu_info_from_env` | `isdigit()`-only → UUID mask = fail-closed (empty) | accept `GPU-*`/hex tokens; index identity becomes list position |
| `fastvideo.rs select_visible_gpu_archs` | `parse::<usize>()` → UUID mask selects nothing | non-numeric mask → fall back to all enumerated archs (mask selects devices, not archs) |
| `env_validation.rs` | value-opaque (`parse_env_file` returns raw values; no numeric grammar constrains mask shape) | **DESCOPED** — nothing to extend; mask values are already unconstrained pass-through |
| `vllm_multi.rs` launcher | honors operator HIP mask verbatim / emits its own | passthrough only — no change |
| `permanent_env.rs` | delegates to canonical generator | inherits fix via same inputs |
| `comfyui/textgen/megatron/freetoken` embedded masks | per-component, pre-existing behavior | out of scope (separate tickets); not made worse |

## 5. Tasks

- **T1 `gpu.rs`:** fix `rocm_index` doc (compact GPU ordinals ≠ agent headers);
  add `uuid: Option<String>` to `AmdGpu` (sysfs `unique_id`, rocminfo fallback);
  add `runtime_gpu_ordinal` scan (GPU-type agents, compacted, iGPU counted) →
  populate on discrete matches; pci_slot matching for rocminfo where names lie.
- **T2 mask builder:** `runtime_device_mask` + unit tests: apu-first, dgpu-first,
  vfio-removal, mixed-uuid, no-rocminfo, single-dGPU+APU (the classic leak).
- **T3 `installer.rs ensure_mlstack_env`:** replace the `rocm_index` join with
  `runtime_device_mask` in merge + regenerate (+ fish mirror); comment surgery.
- **T4 consumers:** benchmarks `_gpu_info_from_env` + `_parse_visible_gpu_indices`
  (token-verbatim) + llama-cpp metric keys (enumerate position — the
  `_gpu{N}` suffix is a numeric contract with widgets/benchmarks_page.rs) +
  `_vllm` visible-devices parser, fastvideo `select_visible_gpu_archs`,
  rccl test. env_validation: descoped (see §4).
- **T5 gates:** RAM precheck (no rust-analyzer / mlstack python; >20G avail),
  `pkill -9 -f "cargo build|rustc"` first; `cargo check` + focused standalone
  tests (never bare `cargo test`), fmt/clippy on touched files.
- **T6 live validation (this box):** regenerate env via the binary path; probe
  `torch.cuda.device_count()==2`, names = [XTX, XT], iGPU absent under
  simulated APU-first ordinal space (unit-level), stale-UUID fail-closed.
- **T7 operator runbook:** enable-iGPU checklist (re-run refresh → verify mask
  shape), rollback note (env.bak timestamps already in place).

## 6. Explicitly out of scope
- vfio/passthrough scripts themselves (user-owned tooling).
- Per-component hardcoded masks listed in §4 (pre-existing, separately ticketed).
- Windows/other-platform generators (Linux ROCm only).
