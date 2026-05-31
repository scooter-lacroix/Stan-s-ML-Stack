# Rusty Llama Hardening & Major Version Release — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend CUDA isolation guards to the Rusty Llama installer, sync upstream llama.cpp HIP improvements, and cut the Anagami major version release with updated docs.

**Architecture:** The llama_cpp.rs installer constructs CMake commands and executes them natively. Guards are added as pre-build validation (CUDA toolkit warning), cmake flag hardening (`-DGGML_CUDA=OFF`), cmake cache assertion, and post-build linkage verification. Upstream sync is done as a git cherry-pick workflow in the Fork repo.

**Tech Stack:** Rust (rusty-stack crate), CMake (llama.cpp build), Git (upstream sync), Markdown (docs)

**Spec Bible:** `docs/superpowers/plans/2026-05-30-rusty-llama-hardening-spec.md`
**Handoff:** `docs/superpowers/plans/2026-05-30-handoff.md`

---

## Task 1: Harden cmake_flags() with Explicit CUDA/Vulkan/Metal OFF

**Files:**
- Modify: `rusty-stack/src/installers/components/llama_cpp.rs`

**Blocking:** None (first task)
**Blocked by:** Nothing

- [ ] **Step 1: Write failing tests for explicit OFF flags**

Add these tests to the existing `#[cfg(test)] mod tests` block in `llama_cpp.rs`:

```rust
#[test]
fn test_cmake_flags_always_disable_cuda() {
    for channel in &["legacy", "stable", "latest"] {
        let config = LlamaCppConfig {
            gpu_arch: "gfx1100".to_string(),
            channel: channel.to_string(),
            ..Default::default()
        };
        let installer = LlamaCppInstaller::new(config);
        let flags = installer.cmake_flags();
        assert!(
            flags.contains(&"-DGGML_CUDA=OFF".to_string()),
            "Channel '{}' must explicitly set GGML_CUDA=OFF",
            channel
        );
    }
}

#[test]
fn test_cmake_flags_always_disable_vulkan() {
    let config = LlamaCppConfig {
        gpu_arch: "gfx1100".to_string(),
        channel: "latest".to_string(),
        ..Default::default()
    };
    let installer = LlamaCppInstaller::new(config);
    let flags = installer.cmake_flags();
    assert!(flags.contains(&"-DGGML_VULKAN=OFF".to_string()));
}

#[test]
fn test_cmake_flags_always_disable_metal() {
    let config = LlamaCppConfig {
        gpu_arch: "gfx1100".to_string(),
        channel: "latest".to_string(),
        ..Default::default()
    };
    let installer = LlamaCppInstaller::new(config);
    let flags = installer.cmake_flags();
    assert!(flags.contains(&"-DGGML_METAL=OFF".to_string()));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib test_cmake_flags_always_disable -p rusty-stack`
Expected: FAIL — flags not yet added

- [ ] **Step 3: Add OFF flags to cmake_flags()**

In `llama_cpp.rs`, locate the `cmake_flags()` method and add three lines after the existing flags:

```rust
pub fn cmake_flags(&self) -> Vec<String> {
    let hip_archs = HipArchs::from_gpu_arch(&self.config.gpu_arch);
    let channel = RocmChannel::from_str(&self.config.channel);
    let gpu_targets = hip_archs.gpu_targets_for_channel(channel.label());

    let mut flags = vec![
        "-DGGML_HIP=ON".to_string(),
        format!("-DGPU_TARGETS={}", gpu_targets),
        // VAL-GUARD-001: Explicitly disable non-ROCm backends to prevent contamination
        "-DGGML_CUDA=OFF".to_string(),
        "-DGGML_VULKAN=OFF".to_string(),
        "-DGGML_METAL=OFF".to_string(),
    ];

    // RDNA3 probes and WMMA flash attention are enabled for stable/latest only
    if channel.enable_wmma_fa() {
        flags.push("-DGGML_HIP_RDNA3_PROBES=ON".to_string());
        flags.push("-DGGML_HIP_ROCWMMA_FATTN=ON".to_string());
    }

    flags
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib test_cmake_flags_always_disable -p rusty-stack`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cargo test -p rusty-stack`
Expected: All 1482+ tests pass, 0 failures

- [ ] **Step 6: Update handoff.md — mark Task 1 complete**

---

## Task 2: Add Pre-Build CUDA Toolkit Warning

**Files:**
- Modify: `rusty-stack/src/installers/components/llama_cpp.rs`

**Blocking:** None
**Blocked by:** Task 1

- [ ] **Step 1: Write the CUDA toolkit detection function**

Add this function to `llama_cpp.rs` above the `impl LlamaCppInstaller` block:

```rust
/// Check if NVIDIA CUDA toolkit artifacts are present on the system.
///
/// Checks for `nvcc` and `nvidia-smi` binaries. This is informational —
/// the build will proceed with HIP regardless, but the user is warned.
///
/// # Validation
///
/// - **VAL-GUARD-002**: Pre-build CUDA toolkit detection emits warning
fn detect_cuda_toolkit_presence() -> Vec<String> {
    let mut found = Vec::new();

    let cuda_indicators = [
        ("nvcc", "NVIDIA CUDA compiler"),
        ("nvidia-smi", "NVIDIA system management interface"),
    ];

    for (binary, description) in &cuda_indicators {
        if std::process::Command::new("which")
            .arg(binary)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            found.push(format!("{} ({})", binary, description));
        }
    }

    // Check for CUDA toolkit directory
    if std::path::Path::new("/usr/local/cuda").exists() {
        found.push("/usr/local/cuda directory".to_string());
    }

    found
}
```

- [ ] **Step 2: Write test for the detection function**

```rust
#[test]
fn test_detect_cuda_toolkit_no_panic() {
    // Should not panic regardless of whether CUDA is installed
    let result = detect_cuda_toolkit_presence();
    // Result is a Vec<String> — may be empty or populated
    assert!(result.len() <= 3); // at most nvcc + nvidia-smi + /usr/local/cuda
}
```

- [ ] **Step 3: Integrate warning into run_source_install()**

In `run_source_install()`, add at the very beginning before the command loop:

```rust
pub fn run_source_install(&self, home: &str) -> Result<(), String> {
    // VAL-GUARD-002: Warn if CUDA toolkit is detected — build uses HIP exclusively
    let cuda_artifacts = detect_cuda_toolkit_presence();
    if !cuda_artifacts.is_empty() {
        log_warn(&format!(
            "NVIDIA CUDA toolkit detected on this system ({}). \
             Rusty Llama builds exclusively with ROCm/HIP — CUDA will NOT be used. \
             This is expected on systems with both AMD and NVIDIA GPUs.",
            cuda_artifacts.join(", ")
        ));
    }

    let commands = self.build_commands(home);
    // ... rest of existing code
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p rusty-stack`
Expected: All tests pass

- [ ] **Step 5: Update handoff.md — mark Task 2 complete**

---

## Task 3: Add CMake Cache Validation

**Files:**
- Modify: `rusty-stack/src/installers/components/llama_cpp.rs`

**Blocking:** None
**Blocked by:** Task 1

- [ ] **Step 1: Write the validate_cmake_cache function**

```rust
/// Validate the generated CMakeCache.txt after cmake configure.
///
/// Asserts that the HIP backend is ON, CUDA is OFF, and GPU_TARGETS
/// matches the expected value. Returns Ok(()) if valid, Err with
/// a descriptive message if any assertion fails.
///
/// # Validation
///
/// - **VAL-GUARD-003**: CMake cache validation asserts correct backend flags
fn validate_cmake_cache(
    cache_path: &std::path::Path,
    expected_gpu_targets: &str,
) -> Result<(), String> {
    let contents = std::fs::read_to_string(cache_path)
        .map_err(|e| format!("Failed to read CMakeCache.txt at {}: {}", cache_path.display(), e))?;

    let mut hip_on = false;
    let mut cuda_off = false;
    let mut gpu_targets_match = false;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }

        if trimmed == "GGML_HIP:BOOL=ON" {
            hip_on = true;
        }
        if trimmed == "GGML_CUDA:BOOL=OFF" {
            cuda_off = true;
        }
        if trimmed.starts_with("GPU_TARGETS:STRING=") {
            let value = trimmed.trim_start_matches("GPU_TARGETS:STRING=");
            gpu_targets_match = value == expected_gpu_targets;
            if !gpu_targets_match {
                return Err(format!(
                    "GPU_TARGETS mismatch: expected '{}', found '{}'",
                    expected_gpu_targets, value
                ));
            }
        }
    }

    if !hip_on {
        return Err("GGML_HIP is not ON in CMakeCache.txt — build is not using ROCm".to_string());
    }
    if !cuda_off {
        return Err("GGML_CUDA is not OFF in CMakeCache.txt — CUDA contamination risk".to_string());
    }

    Ok(())
}
```

- [ ] **Step 2: Write tests for cmake cache validation**

```rust
#[test]
fn test_validate_cmake_cache_valid() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        tmp.path(),
        "# CMakeCache\nGGML_HIP:BOOL=ON\nGGML_CUDA:BOOL=OFF\nGPU_TARGETS:STRING=gfx1030;gfx1100;gfx1101\n",
    )
    .unwrap();
    assert!(validate_cmake_cache(tmp.path(), "gfx1030;gfx1100;gfx1101").is_ok());
}

#[test]
fn test_validate_cmake_cache_cuda_on() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        tmp.path(),
        "GGML_HIP:BOOL=ON\nGGML_CUDA:BOOL=ON\nGPU_TARGETS:STRING=gfx1100\n",
    )
    .unwrap();
    let result = validate_cmake_cache(tmp.path(), "gfx1100");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("CUDA"));
}

#[test]
fn test_validate_cmake_cache_hip_off() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        tmp.path(),
        "GGML_HIP:BOOL=OFF\nGGML_CUDA:BOOL=OFF\nGPU_TARGETS:STRING=gfx1100\n",
    )
    .unwrap();
    let result = validate_cmake_cache(tmp.path(), "gfx1100");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("GGML_HIP"));
}

#[test]
fn test_validate_cmake_cache_gpu_mismatch() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        tmp.path(),
        "GGML_HIP:BOOL=ON\nGGML_CUDA:BOOL=OFF\nGPU_TARGETS:STRING=gfx900\n",
    )
    .unwrap();
    let result = validate_cmake_cache(tmp.path(), "gfx1100");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("mismatch"));
}

#[test]
fn test_validate_cmake_cache_missing_file() {
    let result = validate_cmake_cache(
        std::path::Path::new("/nonexistent/CMakeCache.txt"),
        "gfx1100",
    );
    assert!(result.is_err());
}
```

- [ ] **Step 3: Integrate into run_source_install() after cmake configure**

In `run_source_install()`, after the cmake configure command succeeds (index 3 in the commands list), add cache validation:

```rust
for (idx, command) in commands.iter().enumerate() {
    // ... existing execution code ...

    // VAL-GUARD-003: After cmake configure, validate the cache
    if idx == 3 {
        // cmake configure just ran — validate the cache
        let cache_path = std::path::PathBuf::from(&command.working_dir.as_ref().unwrap_or(&PathBuf::from(".")))
            .join("build")
            .join("CMakeCache.txt");
        let expected_targets = self.cmake_flags()
            .iter()
            .find(|f| f.starts_with("-DGPU_TARGETS="))
            .map(|f| f.trim_start_matches("-DGPU_TARGETS=").to_string())
            .unwrap_or_default();
        if let Err(e) = validate_cmake_cache(&cache_path, &expected_targets) {
            return Err(format!("CMake cache validation failed: {}", e));
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p rusty-stack`
Expected: All tests pass

- [ ] **Step 5: Update handoff.md — mark Task 3 complete**

---

## Task 4: Add Post-Build ROCm Linkage Verification

**Files:**
- Modify: `rusty-stack/src/installers/components/llama_cpp.rs`

**Blocking:** None
**Blocked by:** Task 1

- [ ] **Step 1: Write the linkage enforcement function**

```rust
/// Verify that the installed binary has ROCm/HIP linkage and NOT CUDA linkage.
///
/// Uses `ldd` to check for:
/// - REQUIRED: `amdhip64` or `hipblas` or `libhip` (ROCm present)
/// - FORBIDDEN: `libcuda` or `libnvidia` or `libcudart` (CUDA contamination)
///
/// # Validation
///
/// - **VAL-GUARD-004**: Post-build linkage verification rejects CUDA-linked binaries
/// - **VAL-GUARD-005**: Post-build linkage verification requires ROCm linkage
fn verify_binary_linkage(binary_path: &std::path::Path) -> Result<(), String> {
    if !binary_path.exists() {
        return Err(format!(
            "Binary not found at {} — install may have failed",
            binary_path.display()
        ));
    }

    let output = std::process::Command::new("ldd")
        .arg(binary_path)
        .output()
        .map_err(|e| format!("Failed to run ldd on {}: {}", binary_path.display(), e))?;

    if !output.status.success() {
        return Err(format!(
            "ldd failed on {}: {}",
            binary_path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let ldd_output = String::from_utf8_lossy(&output.stdout);

    // Check for CUDA contamination
    let cuda_markers = ["libcuda.so", "libnvidia", "libcudart"];
    for marker in &cuda_markers {
        if ldd_output.contains(marker) {
            return Err(format!(
                "CUDA contamination detected: {} links against '{}'. \
                 The binary was built with CUDA instead of ROCm/HIP. \
                 This should not happen — check CMake configuration.",
                binary_path.display(),
                marker
            ));
        }
    }

    // Check for ROCm linkage
    let has_rocm = ldd_output.contains("amdhip64")
        || ldd_output.contains("hipblas")
        || ldd_output.contains("libhip");

    if !has_rocm {
        return Err(format!(
            "ROCm/HIP linkage not found in {}. The binary may be a CPU-only build \
             or was built without GPU acceleration. Expected linkage to amdhip64 or hipblas.",
            binary_path.display()
        ));
    }

    Ok(())
}
```

- [ ] **Step 2: Write tests**

```rust
#[test]
fn test_verify_binary_linkage_missing_binary() {
    let result = verify_binary_linkage(std::path::Path::new("/nonexistent/binary"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[test]
fn test_verify_binary_linkage_no_panic() {
    // Test with a known binary (bash) — should fail on ROCm linkage check
    // but should NOT panic
    let result = verify_binary_linkage(std::path::Path::new("/usr/bin/bash"));
    // bash won't have ROCm linkage, so this should be an Err
    assert!(result.is_err());
}
```

- [ ] **Step 3: Integrate into install() after successful build/download**

In the `install()` method, after both the prebuilt download path and source build path succeed, add linkage verification:

```rust
// VAL-GUARD-004, VAL-GUARD-005: Verify binary linkage after install
let binary_path = self.config.detection_binary_path(home);
if binary_path.exists() {
    if let Err(e) = verify_binary_linkage(&binary_path) {
        log_warn(&format!("Binary linkage verification: {}", e));
        // Don't fail the install — the binary may still work
        // but log clearly for troubleshooting
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p rusty-stack`
Expected: All tests pass

- [ ] **Step 5: Update handoff.md — mark Task 4 complete**

---

## Task 5: Upstream Sync in Fork Repo

**Files:**
- Work in: `Fork/llama.cpp-turboquant-hip/`

**Blocking:** None (independent of Rust tasks)
**Blocked by:** Nothing

- [ ] **Step 1: Add upstream remote**

```bash
cd Fork/llama.cpp-turboquant-hip
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream --depth=100
```

- [ ] **Step 2: Create sync branch**

```bash
git checkout -b sync/upstream-2026-05
```

- [ ] **Step 3: Cherry-pick upstream HIP improvements**

Cherry-pick in dependency order. For each commit, resolve conflicts preserving local RDNA2/3 work:

```bash
# Mar 12: HIP debug build -O2 fix
git cherry-pick d63aa39 || git cherry-pick --continue

# Mar 19: Windows RDNA compiler bug fix
git cherry-pick 3408072 || git cherry-pick --continue

# Mar 22: BF16 flash attention vec kernel
git cherry-pick db9d8aa || git cherry-pick --continue

# Apr 9: Backend-agnostic tensor parallelism
git cherry-pick d6f3030 || git cherry-pick --continue
```

**Conflict resolution rules:**
- If conflict is in `ggml/src/ggml-hip/` → keep LOCAL (our RDNA2/3 probes/kernels)
- If conflict is in `ggml/CMakeLists.txt` → merge both sides, keep our RDNA3 options
- If conflict is in `src/` or `common/` → prefer UPSTREAM (general improvements)
- If conflict is in `tools/` → prefer UPSTREAM

- [ ] **Step 4: Build and verify**

```bash
cmake -B build-sync -DGGML_HIP=ON -DGGML_CUDA=OFF -DGPU_TARGETS="gfx1030;gfx1100;gfx1101" -DGGML_HIP_RDNA3_PROBES=ON
cmake --build build-sync -j$(nproc) --target llama-cli llama-bench llama-server
./build-sync/bin/llama-cli --help
```

Expected: Clean compile, `--help` exits 0

- [ ] **Step 5: Run hygiene validation**

```bash
./scripts/validate_hygiene.sh
```

Expected: All checks pass (exit 0)

- [ ] **Step 6: Update handoff.md — mark Task 5 complete**

---

## Task 6: Update CHANGELOG.md for Anagami Release

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `VERSION`

**Blocking:** Tasks 1-5 (must be done last)
**Blocked by:** Tasks 1, 2, 3, 4, 5

- [ ] **Step 1: Update VERSION file**

```bash
echo "0.2.0" > VERSION
```

- [ ] **Step 2: Add Anagami release header to CHANGELOG.md**

Replace `## [Unreleased]` with `## [0.2.0] - 2026-05-30 (Anagami)` and add a new empty `## [Unreleased]` above it. Add a release summary section at the top of the Anagami entry:

```markdown
## [Unreleased]

## [0.2.0] - 2026-05-30 (Anagami)

### Release Highlights

- **All-Rust Installer**: All 35 installer components are now native Rust — zero shell script dependencies for the install path.
- **Rusty Llama Integration**: llama.cpp-turboquant-hip with RDNA2/3/4 support, prebuilt binaries, source compile fallback, and CUDA isolation guards.
- **CUDA Isolation Hardening**: Explicit `-DGGML_CUDA=OFF` enforcement, CMake cache validation, and post-build binary linkage verification.
- **Upstream Sync**: Merged upstream llama.cpp BF16 flash attention, tensor parallelism, and Windows HIP fixes.
- **Windows Alpha**: Initial Windows x86_64 builds with WSL2 support. Actively seeking testers.

### Mission: Rusty Llama Integration (2026-04 — 2026-05)
```

Keep all existing entries under `### Mission: Rusty Llama Integration` — they become part of the Anagami release.

- [ ] **Step 3: Update handoff.md — mark Task 6 complete**

---

## Task 7: Update README.md with Rusty Llama and Windows Highlights

**Files:**
- Modify: `README.md`

**Blocking:** None
**Blocked by:** Tasks 1-5

- [ ] **Step 1: Read current README.md**

Read the file first to understand its current structure.

- [ ] **Step 2: Add Rusty Llama section**

Add a section highlighting Rusty Llama as a first-class component, including:
- CUDA isolation guarantee
- RDNA2/3/4 channel-aware builds
- Prebuilt binary + source compile strategy
- Link to Fork README for benchmarks

- [ ] **Step 3: Add Windows Alpha section**

Add a section noting Windows Alpha status with a call-to-action for testers.

- [ ] **Step 4: Update architecture diagram**

Replace any references to shell script backend with the all-Rust native path. Show the installer flow:

```
User → Rusty Stack TUI → Native Rust Installer Modules (35 components)
                            ↓
                    Component-specific logic:
                    - PyTorch: pip install from ROCm index
                    - Rusty Llama: CMake source build with CUDA guards
                    - vLLM: git clone + filtered pip install
                    - ...
```

- [ ] **Step 5: Update handoff.md — mark Task 7 complete**

---

## Task 8: Update INSTALLER_STATUS.md

**Files:**
- Modify: `docs/INSTALLER_STATUS.md`

**Blocking:** None
**Blocked by:** Tasks 1-5

- [ ] **Step 1: Replace architecture diagram**

Replace the shell-scripts-based diagram with:

```markdown
## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    User-Facing Interface                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐                                        │
│  │   Rusty-Stack TUI   │  ← Primary (Rust + Ratatui)            │
│  └──────────┬──────────┘                                        │
└─────────────┼───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│           Native Rust Installer Modules (35 components)          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │  ROCm    │ │ PyTorch  │ │  vLLM    │ │ Rusty Llama      │   │
│  │ rocm.rs  │ │pytorch.rs│ │ vllm.rs  │ │ llama_cpp.rs     │   │
│  └──────────┘ └──────────┘ └──────────┘ │ + CUDA Guards    │   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │ + CMake Cache    │   │
│  │ Triton   │ │DeepSpeed │ │ FlashAttn│ │   Validation     │   │
│  └──────────┘ └──────────┘ └──────────┘ │ + Linkage Verify │   │
│  ... (35 total, all native Rust)        └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```
```

- [ ] **Step 2: Update component status table**

Mark all components as `Native Rust` instead of `Shell Script`. Add llama-cpp row.

- [ ] **Step 3: Update handoff.md — mark Task 8 complete**

---

## Task 9: Update Fork README with Upstream Sync and CUDA Isolation Notes

**Files:**
- Modify: `Fork/llama.cpp-turboquant-hip/README.md`

**Blocking:** Task 5
**Blocked by:** Task 5

- [ ] **Step 1: Add upstream sync note**

Add after the Installation section:

```markdown
## Upstream Sync

This fork is synced with upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) as of May 2026. Key upstream improvements included:

- Native BF16 flash attention for vec kernel (RDNA3 performance uplift)
- Backend-agnostic tensor parallelism (multi-GPU)
- Windows HIP compiler bug fix
- HIP debug build optimization

Local additions preserved:
- RDNA2 experimental kernels (BFE dequant, async pipeline, MoE prefill)
- RDNA3 WMMA probes and validation
- TurboQuant optimized quantization
```

- [ ] **Step 2: Add CUDA isolation note**

```markdown
## CUDA Isolation

When installed via Rusty Stack, Rusty Llama enforces strict CUDA isolation:

- `-DGGML_CUDA=OFF` is always set explicitly
- CMake cache is validated post-configure
- Built binary is verified for ROCm/HIP linkage (no CUDA `.so` contamination)
- NVIDIA toolkit presence triggers an informational warning (build proceeds with HIP)

This ensures a clean AMD-only binary regardless of what else is installed on the system.
```

- [ ] **Step 3: Update handoff.md — mark Task 9 complete**

---

## Dependency Graph

```diagram
╭──────────╮
│  Task 1  │ ← cmake_flags() hardening (FIRST)
╰────┬─────╯
     │ blocks
     ├──────────────────────────────────────────╮
     ▼                ▼                ▼        │
╭──────────╮  ╭──────────╮  ╭──────────╮       │
│  Task 2  │  │  Task 3  │  │  Task 4  │       │
│ CUDA warn│  │ CMake val│  │ Linkage  │       │
╰──────────╯  ╰──────────╯  ╰──────────╯       │
                                                │
╭──────────╮ (independent)                      │
│  Task 5  │ ← upstream sync                    │
╰────┬─────╯                                    │
     │ blocks                                   │
     ▼                                          │
╭──────────╮                                    │
│  Task 9  │ ← Fork README                     │
╰──────────╯                                    │
                                                │
     All tasks 1-5 block:                       │
     ▼           ▼           ▼                  │
╭──────────╮ ╭──────────╮ ╭──────────╮         │
│  Task 6  │ │  Task 7  │ │  Task 8  │         │
│CHANGELOG │ │ README   │ │INST_STAT │         │
╰──────────╯ ╰──────────╯ ╰──────────╯
```

**Parallel execution:** Tasks 2, 3, 4 can run in parallel after Task 1. Task 5 is fully independent. Tasks 6, 7, 8, 9 are doc tasks that can run in parallel after implementation tasks complete.
