# FastVideo ROCm-Native Dependency-Safe Repair Design

**Date:** 2026-07-17
**Status:** Implemented; user installation/runtime validation pending
**FastVideo base:** `scooter-lacroix/FastVideo@472f6fb712fdcc79cfeb5fe9a2533c51b148d361`
**FastVideo pinned repair:** `scooter-lacroix/FastVideo@22e448771ebf5c81108f1c57b9c7ef4d5c26d182`
**FastVideo work branch:** `feature/rocm-native-no-cutlass`

## Goal

Make FastVideo install and run on the managed Rusty ROCm environment without resolving, installing, upgrading, or replacing component dependencies and without retaining NVIDIA/CUDA packages, indexes, CUTLASS, or ThunderKittens source.

## Ownership boundary

The FastVideo fork owns its ROCm-native kernel implementation and dependency-free metadata. Rusty Stack owns environment provisioning, prerequisite validation, exact source pinning, offline installation, contamination enforcement, and end-to-end verification.

Rusty never repairs missing FastVideo prerequisites during the component install. It fails before mutation and directs the user to the Rusty component that owns the missing prerequisite.

## FastVideo fork design

The fork becomes ROCm-only for this branch:

- Delete CUTLASS and ThunderKittens gitlinks and `.gitmodules`.
- Remove CUDA wheel indexes, NVIDIA classifiers, and runtime dependency resolution metadata.
- Use CMake `LANGUAGES CXX HIP`; mark HIP sources explicitly.
- Remove bundled FlashAttention C++ sources and bindings. Python uses the Rusty-managed `flash_attn`.
- Preserve the Python extension and binding contracts: `quant_cuda`, `gemm_cuda`, `rms_norm_cuda`, and `layer_norm_cuda`.

Quantization and normalization use HIP runtime types, explicit saturation, and HIP reductions. The NVIDIA CuTe GEMM is replaced with a correctness-first ATen implementation that performs blockwise INT8 matrix multiplication against the installed ROCm PyTorch. It preserves the existing scale layout and output contract. A later CK/hipBLASLt optimization may replace it without changing Python bindings.

## Rusty installer design

Rusty pins the exact pushed fork commit and checks out detached. It never initializes submodules.

The install sequence is:

1. Remove any prior build tree, clone fresh, checkout detached, and verify exact commit plus a pristine tracked/untracked worktree.
2. Apply no source mutation beyond fail-closed policy validation.
3. Run a read-only prerequisite preflight for ROCm Torch, managed FlashAttention, scikit-build-core, setuptools, wheel, CMake, Ninja, HIP compiler, and required ROCm headers.
4. Snapshot installed distributions and reject NVIDIA/CUDA contamination using Rusty’s canonical runtime-prefix policy.
5. Build/install the kernel and root package locally with `--no-deps --no-build-isolation --no-index`, `PIP_NO_INDEX=1`, and `PIP_DISABLE_PIP_VERSION_CHECK=1`.
6. Compare the distribution snapshot. Only `fastvideo` and `fastvideo-kernel` may change.
7. Run the existing compiled-op smoke on each visible logical GPU.

The FlashAttention preflight imports public APIs from `flash_attn` and private VMoBA APIs from `flash_attn.flash_attn_interface`.

The mutating FastVideo `BuildDeps` step is removed entirely.

## Managed environment restoration

FastVideo must not repair the damage it caused. Rusty’s PyTorch/environment owner provisions exact versions with `--no-deps`: `setuptools==81.0.0`, `cmake==4.3.4`, `scikit-build-core==1.0.3`, `pathspec==1.1.1`, `wheel==0.47.0`, `ninja==1.13.0`, `imageio==2.36.0`, `diffusers==0.33.1`, `remote-pdb==2.1.0`, `ftfy==6.3.1`, and `wcwidth==0.2.13`. FastVideo only verifies these pins.

The repair validates `pip check` and reports pre-existing vLLM inconsistencies separately from FastVideo-caused changes.

## Source policy

The pinned fork must contain none of:

- Git submodules or gitlinks.
- CUTLASS, CuTe, or ThunderKittens includes/symbols.
- NVIDIA/CUDA package requirements or wheel indexes.
- CUDA runtime linkage.

The policy scan allows `torch.cuda` and ATen CUDA-compatibility namespaces because ROCm PyTorch exposes those APIs.

## Error handling

Every policy or prerequisite failure occurs before package installation and names the owning Rusty repair path. Commit mismatch, source-policy drift, dependency snapshot drift, and NVIDIA contamination are fatal.

## Verification

- TDD red/green tests for fork source policy and numeric contracts.
- Kernel build against the existing managed ROCm Torch with network disabled.
- FP16/BF16 quantization, block-scaled GEMM, RMSNorm, and LayerNorm comparisons.
- Separate GPU runs under `ROCR_VISIBLE_DEVICES=0` and `1`.
- Wheel/readelf scan for CUDA/NVIDIA libraries.
- Rust focused standalone tests; never `cargo test`.
- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, and CPU-limited release build.
- Atomic installed-binary replacement with matching checksums.
