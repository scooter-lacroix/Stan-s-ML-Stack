# FastVideo ROCm-Native Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace FastVideo’s NVIDIA CUTLASS/ThunderKittens kernel dependency with ROCm-native code and make Rusty’s FastVideo installation fully offline, dependency-preserving, pinned, and verifiable.

**Architecture:** The external FastVideo fork owns a ROCm-only kernel branch and preserves existing Python bindings. Rusty pins the pushed commit, performs read-only prerequisite and source-policy checks, installs local packages with network/dependency resolution disabled, and verifies that no managed distribution changed except FastVideo itself.

**Tech Stack:** Rust, Python 3.12, PyTorch 2.12 ROCm 7.2, C++20, HIP, CMake/scikit-build-core, ATen.

**Execution status:** Tasks 1-10 are complete. Fork repair is pushed at `22e448771ebf5c81108f1c57b9c7ef4d5c26d182`; Rust code is committed at `5beed1c`; 26 focused tests, formatting, clippy with warnings denied, and the exact staged release build are green. The installed `rusty-stack 0.3.1` checksum is `18007be04bd194b94562220532ac6c983aef1319aacb9b694fe9469099ac2130`. Task 11 FastVideo installation/runtime validation is intentionally reserved for the user.

---

### Task 1: Establish fork policy and API contract tests

**Files:**
- Create: `/tmp/FastVideo_ROCm_design_20260717/fastvideo-kernel/tests/test_rocm_source_policy.py`
- Modify: `/tmp/FastVideo_ROCm_design_20260717/fastvideo-kernel/tests/test_turbodiffusion.py`

- [ ] **Step 1: Write failing source-policy tests**

Add tests that walk tracked source and metadata while excluding `.git` and assert:

- no `.gitmodules` or gitlink entries;
- no path named `include/cutlass` or `include/tk`;
- no CUTLASS/CuTe/TK include or symbol tokens;
- no NVIDIA/CUDA wheel indexes or `nvidia-*` requirements;
- CMake declares `LANGUAGES CXX HIP`;
- active source list contains no FlashAttention C++ source.

Allow `torch.cuda`, `ATen/cuda`, and `c10/cuda` compatibility namespaces.

- [ ] **Step 2: Write failing API contract tests**

Assert the Python wrapper still exports `quant_cuda`, `gemm_cuda`, `rms_norm_cuda`, and `layer_norm_cuda`, and document current argument/output shapes.

- [ ] **Step 3: Run RED**

Run:

```bash
/home/scooter/.mlstack/global/bin/python -m pytest   fastvideo-kernel/tests/test_rocm_source_policy.py   fastvideo-kernel/tests/test_turbodiffusion.py -q
```

Expected: source policy fails on gitlinks/CUTLASS/TK/CUDA metadata; existing numeric tests may fail because the extension is not installed.

### Task 2: Remove fork metadata and submodules

**Files:**
- Delete: `/tmp/FastVideo_ROCm_design_20260717/.gitmodules`
- Delete gitlinks: `/tmp/FastVideo_ROCm_design_20260717/fastvideo-kernel/include/cutlass`
- Delete gitlinks: `/tmp/FastVideo_ROCm_design_20260717/fastvideo-kernel/include/tk`
- Modify: `/tmp/FastVideo_ROCm_design_20260717/pyproject.toml`
- Modify: `/tmp/FastVideo_ROCm_design_20260717/fastvideo-kernel/pyproject.toml`

- [ ] **Step 1: Remove gitlinks and CUDA indexes**

Use Git index operations to remove the two gitlinks and `.gitmodules`. Remove `pytorch-cu128` and CUDA source selection from root metadata.

- [ ] **Step 2: Make runtime metadata dependency-free**

Remove fork runtime dependency resolution from both project metadata files for this Rusty-specific branch. Keep build backend declarations so `--no-build-isolation` uses already-managed tools.

- [ ] **Step 3: Remove NVIDIA classifiers/descriptions**

Change kernel description/classifier to ROCm/HIP and ensure no package metadata advertises CUDA requirements.

- [ ] **Step 4: Run focused policy test**

Expected: metadata/submodule assertions pass; source CUTLASS assertions remain RED.

### Task 3: Replace common, quantization, and normalization CUTLASS usage

**Files:**
- Create: `/tmp/FastVideo_ROCm_design_20260717/fastvideo-kernel/csrc/hip_native/compat.hpp`
- Modify: `fastvideo-kernel/csrc/turbodiffusion/common/common.hpp`
- Modify: `fastvideo-kernel/csrc/turbodiffusion/common/launch.hpp`
- Modify: `fastvideo-kernel/csrc/turbodiffusion/common/load.hpp`
- Modify: `fastvideo-kernel/csrc/turbodiffusion/common/store.hpp`
- Modify: `fastvideo-kernel/csrc/turbodiffusion/quant/quant.cu`
- Modify: `fastvideo-kernel/csrc/turbodiffusion/quant/quant.hpp`
- Modify: `fastvideo-kernel/csrc/turbodiffusion/norm/rmsnorm.cu`
- Modify: `fastvideo-kernel/csrc/turbodiffusion/norm/rmsnorm.hpp`
- Modify: `fastvideo-kernel/csrc/turbodiffusion/norm/layernorm.cu`
- Modify: `fastvideo-kernel/csrc/turbodiffusion/norm/layernorm.hpp`

- [ ] **Step 1: Add failing numeric contract tests**

For FP16 and BF16, compare quantized tensors/scales, RMSNorm, and LayerNorm against PyTorch references. Include shapes with tails rather than exact 128 multiples.

- [ ] **Step 2: Run RED**

Expected: current extension/build cannot satisfy source policy and tail contracts.

- [ ] **Step 3: Add HIP compatibility header**

Define HIP runtime types, `FV_HOST_DEVICE`, `FV_DEVICE`, unroll annotation, managed half/BF16 types, and explicit saturating nearest INT8 conversion.

- [ ] **Step 4: Replace CUTLASS aliases/macros**

Replace every active non-GEMM CUTLASS type/macro/include. Replace CUDA runtime calls with HIP equivalents, remove `__grid_constant__`, and use wave-safe shuffle masks/width.

- [ ] **Step 5: Run source-policy and compile checks**

Expected: no CUTLASS tokens remain outside GEMM; quant/norm compile under hipcc.

### Task 4: Replace CuTe GEMM with managed ATen ROCm implementation

**Files:**
- Create: `/tmp/FastVideo_ROCm_design_20260717/fastvideo-kernel/csrc/hip_native/gemm_rocm.cpp`
- Modify: `/tmp/FastVideo_ROCm_design_20260717/fastvideo-kernel/csrc/common_extension.cpp`
- Delete: active CuTe implementation files under `fastvideo-kernel/csrc/turbodiffusion/gemm/` after bindings migrate.

- [ ] **Step 1: Write failing GEMM reference tests**

Cover FP16/BF16 output, non-square matrices, K tails, multiple 128x128 scale blocks, and both visible GPUs. Reference:

```python
for k0 in range(0, k, 128):
    partial = a[:, k0:k0+128].float() @ b[:, k0:k0+128].float().T
    partial *= a_scale[:, k0 // 128].repeat_interleave(128)[:m, None]
    partial *= b_scale[:, k0 // 128].repeat_interleave(128)[:n][None, :]
    expected += partial
```

- [ ] **Step 2: Run RED**

Expected: current CuTe source violates policy and cannot compile without CUTLASS.

- [ ] **Step 3: Implement minimal ATen GEMM**

Implement `gemm_cuda` in C++ using installed ATen operations on ROCm tensors. Validate device, dtype, contiguity, rank, dimensions, and scale shapes; accumulate FP32 blockwise and copy/cast into the existing output contract.

- [ ] **Step 4: Remove CuTe GEMM sources**

Delete all reachable CuTe/PTX implementation files after the CMake/binding migration.

- [ ] **Step 5: Run GEMM tests**

Expected: numeric tests pass within declared tolerances on `ROCR_VISIBLE_DEVICES=0` and `1`.

### Task 5: Make fork build ROCm-only and remove bundled FlashAttention

**Files:**
- Modify: `/tmp/FastVideo_ROCm_design_20260717/fastvideo-kernel/CMakeLists.txt`
- Modify: `/tmp/FastVideo_ROCm_design_20260717/fastvideo-kernel/csrc/common_extension.cpp`
- Delete: `fastvideo-kernel/csrc/attention/flash_attn_rocm.cpp` if present.

- [ ] **Step 1: Write failing CMake policy assertions**

Assert explicit HIP language, explicit HIP source properties, no CUDA/TK/CUTLASS branches/includes, and only the ROCm-native source set.

- [ ] **Step 2: Run RED**

Expected: current CMake violates assertions.

- [ ] **Step 3: Implement ROCm-only CMake**

Use `project(... LANGUAGES CXX HIP)`, locate managed Python/Torch, mark `.cu` sources `LANGUAGE HIP`, include only project/ATen/HIP paths, and link only managed Torch/ROCm libraries.

- [ ] **Step 4: Remove FlashAttention bindings**

The Python VMoBA path remains routed to managed `flash_attn`; no C++ FlashAttention symbols remain.

- [ ] **Step 5: Run full fork source-policy suite**

Expected: all policy tests GREEN.

### Task 6: Build and validate the fork offline

**Files:**
- Build artifact only; no new source files expected.

- [ ] **Step 1: Snapshot the managed environment**

Record installed distribution names/versions and confirm no normalized `nvidia-*`/CUDA runtime distributions.

- [ ] **Step 2: Build/install kernel offline**

Run from `fastvideo-kernel`:

```bash
PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 CMAKE_ARGS='-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1101' GPU_ARCHS='gfx1100;gfx1101' MAX_JOBS=4 /home/scooter/.mlstack/global/bin/python -m pip install   --break-system-packages --no-deps --no-build-isolation --no-index .
```

- [ ] **Step 3: Build/install root package offline**

Run the same pip safety flags from the fork root.

- [ ] **Step 4: Run numeric verification per GPU**

Run quant/GEMM/norm tests separately with `ROCR_VISIBLE_DEVICES=0` and `1`. Run managed FlashAttention public/private VMoBA probes.

- [ ] **Step 5: Inspect artifacts**

Use `readelf -d` on the extension and inspect wheel contents. Fail on CUDA/NVIDIA libraries or source names.

- [ ] **Step 6: Compare distribution snapshots**

Only `fastvideo` and `fastvideo-kernel` may differ.

### Task 7: Commit and push the repaired FastVideo fork

**Files:** all fork changes above.

- [ ] **Step 1: Review exact fork diff**

Confirm no generated binaries, build directories, caches, or unrelated files are staged.

- [ ] **Step 2: Commit**

Create focused conventional commits for policy/metadata, HIP common kernels, ATen GEMM, and verification if separation remains coherent.

- [ ] **Step 3: Push**

Push `feature/rocm-native-no-cutlass` to `origin`.

- [ ] **Step 4: Record immutable SHA**

Capture the pushed commit SHA and verify the remote branch resolves to it.

### Task 8: Pin FastVideo SHA and create Rusty work branch

**Files:**
- Modify: `rusty-stack/src/installers/components/fastvideo.rs`

- [ ] **Step 1: Write failing pin tests**

Assert the installer uses the immutable 40-character SHA, has no mutable checkout target, and creates no submodule command.

- [ ] **Step 2: Run focused Rust test RED**

Compile/run `fastvideo.rs` with standalone `rustc --test`; never `cargo test`.

- [ ] **Step 3: Pin pushed SHA**

Replace mutable branch checkout with detached exact commit verification.

- [ ] **Step 4: Create Rusty branch**

Create `fix/fastvideo-rocm-native-dependency-safety` while preserving all existing user changes.

- [ ] **Step 5: Run focused test GREEN**

Expected: pin/no-submodule assertions pass.

### Task 9: Replace Rusty dependency mutation with managed preflight and offline install

**Files:**
- Modify: `rusty-stack/src/installers/components/fastvideo.rs`
- Modify: `rusty-stack/src/installer.rs`
- Modify: `rusty-stack/src/installers/common/nvidia_blocklist.rs`

- [ ] **Step 1: Write failing command-contract tests**

Assert:

- no `BuildDeps` step;
- private FlashAttention imports use `flash_attn.flash_attn_interface`;
- preflight validates `torch.version.hip`, managed marker, build tools, HIP compiler/headers;
- every local pip command has `--no-deps --no-build-isolation --no-index`;
- every local pip command sets `PIP_NO_INDEX=1` and disables pip version checks;
- source policy and distribution snapshot guards run before/after installation.

- [ ] **Step 2: Run RED**

Expected: current `BuildDeps`, wrong import, and missing offline/snapshot guards fail.

- [ ] **Step 3: Implement read-only prerequisite preflight**

Use the selected Python and external tool/header probes. Error messages identify Rusty’s owning component; never invoke pip.

- [ ] **Step 4: Implement source-policy guard**

Reject gitlinks, submodule metadata, CUTLASS/CuTe/TK tokens, CUDA indexes/requirements, and unexpected commit before build.

- [ ] **Step 5: Implement dependency snapshot/contamination guard**

Reuse canonical normalization/blocklist logic. Allow only `fastvideo` and `fastvideo-kernel` changes after installation.

- [ ] **Step 6: Make installs offline**

Add required pip flags/environment and preserve selected managed Python/ROCm visibility variables.

- [ ] **Step 7: Run focused tests GREEN**

Expected: all FastVideo standalone tests pass.

### Task 10: Restore managed environment ownership

**Files:**
- Create: `rusty-stack/src/installers/common/managed_python_build_tools.rs`
- Modify: `rusty-stack/src/installers/common/mod.rs`
- Modify: `rusty-stack/src/installer.rs`
- Test: `rusty-stack/src/installers/common/managed_python_build_tools.rs` with its focused standalone test entry point.

- [ ] **Step 1: Write failing ownership/version tests**

Assert the shared Rusty owner produces a selected-Python pip command with `--no-deps` and exact pins:
`setuptools==81.0.0`, `cmake==4.3.4`, `scikit-build-core==1.0.3`,
`pathspec==1.1.1`, `wheel==0.47.0`, and `ninja==1.13.0`.
Assert it contains no CUDA/NVIDIA index or package tokens.

- [ ] **Step 2: Verify RED**

Compile the new standalone module test and confirm it fails because the managed owner does not yet exist.

- [ ] **Step 3: Implement the shared managed owner**

Build the exact-pin command in `managed_python_build_tools.rs`. Invoke it from the core PyTorch
installation flow immediately after the managed ROCm PyTorch command and before downstream components.
FastVideo must only preflight these packages and must never invoke this owner.

- [ ] **Step 4: Run focused tests GREEN**

Confirm exact pins, selected-Python targeting, `--no-deps`, and absence of NVIDIA/CUDA resolver inputs.

- [ ] **Step 5: Restore the live environment through the owner once**

Run the generated exact-pin selected-Python command once. Do not use a dependency resolver and do not
install or modify Torch, ROCm, Flash Attention, Triton, or any NVIDIA/CUDA package.

- [ ] **Step 6: Run `pip check` and classify output**

Confirm the Torch/setuptools conflict is removed. Report pre-existing vLLM conflicts separately.

### Task 11: Integrate verification, docs, build, and deploy

**Files:**
- Modify: `rusty-stack/src/component_status.rs`
- Modify: `rusty-stack/docs/FASTVIDEO_REMEDIATION_PLAN.md`
- Modify: `rusty-stack/docs/superpowers/specs/2026-07-17-fastvideo-rocm-native-repair-design.md`
- Modify: this implementation plan’s checkboxes.

- [ ] **Step 1: Verify shared GPU smoke**

Keep one shared compiled-op snippet and ensure both component-specific and aggregate verification paths call it under inherited managed visibility masks.

- [ ] **Step 2: Run focused Rust tests**

Compile and run standalone tests for modified component files. Never invoke `cargo test`.

- [ ] **Step 3: Run formatting/lint/build**

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets
nice -n 19 cargo build --release -j 2
```

Expected: zero warnings/errors.

- [ ] **Step 4: User runs installed FastVideo verification**

The user runs the Rusty FastVideo installer, then confirms imports, source/distribution policy checks, and compiled operations on both dGPUs.

- [x] **Step 5: Update remediation evidence**

Record fork SHA, build/test evidence, environment restoration, and any remaining performance-only work.

- [x] **Step 6: Deploy atomically**

Copy to `~/.cargo/bin/rusty-stack.new`, atomically rename, and verify source/deployed SHA-256 equality.
