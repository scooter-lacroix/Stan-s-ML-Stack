# FreeToken (HIP/ROCm) — Extension Guide

**Component:** `freetoken` · **Tier:** Candidate · **Deps:** `pytorch`, `rocm`
**Fork:** `scooter-lacroix/FreeToken`, branch `feature/rocm` (upstreamable commit series; PR to `FlashML-org/FreeToken` planned)

FreeToken is a MoE-offload LLM serving engine: it serves frontier MoE checkpoints
(Qwen3.5-MoE, DeepSeek-V4, GLM, gpt-oss, MiniMax class) on consumer GPUs by keeping
experts in pinned host RAM behind an LRU VRAM slot cache, streaming misses over
PCIe or computing them on the CPU (hybrid mode). It exposes OpenAI- and
Anthropic-compatible HTTP APIs.

## What the ROCm port changes

The port is **feature-detected, not forked**: upstream sources compile on HIP
through guards keyed off `torch.version.hip` / `__HIP_PLATFORM_AMD__`.

| Layer | CUDA path | HIP path |
|---|---|---|
| CppExtensions (`_pinned_tensor`, `_cpu_moe`) | `-lcudart`, `cuda_runtime_api.h` | name shim → `hip/hip_runtime_api.h`, `-lamdhip64`; stream memops dlsym'd from `libamdhip64.so` (`hipStreamWrite/WaitValue64`); `host_ptr_identity` probed functionally |
| tvm-ffi JIT kernels | nvcc, `TVM_FFI_CUDA_ARCH_LIST` gencode | `backend="hip"` → hipcc, `--offload-arch` from `TVM_FFI_ROCM_ARCH_LIST` or rocminfo; nvcc-only flags dropped |
| GGUF kernels (llama.cpp vendored) | nvcc + `-ccbin clang++` | torch cpp_extension in-process hipifier; ROCm 7's 64-bit-mask shuffles handled in `dispatch.h` (`width=32` segmentation) |
| Attention backend | trtllm / fa+fi / fi | auto-resolves to the pure-Triton backend (in-repo kernels) |
| MoE LRU slot cache | flashlib Triton `lru_ensure` | same — runs as-is on gfx1100 |
| NVIDIA-only extras | flashinfer `[fi]`, sgl-kernel `[sgl]`, vLLM Marlin NVFP4, PDL | not installed; every path has an in-repo Triton fallback |

## Install layout (first-class stack integration)

- **Venv:** `~/.mlstack/venvs/freetoken` — created from the managed python
  (`~/.mlstack/global/bin/python`) because FreeToken's pins
  (`transformers>=5.5,<6`, `numpy<2.5`, `huggingface_hub>=1.5`) are ahead of the
  co-installed vllm/megatron/textgen set. Torch inside the venv comes from the
  same ROCm wheel index the core pytorch installer uses.
- **Source clone:** `~/.mlstack/freetoken` (fork, `feature/rocm`)
- **Launcher:** `~/.mlstack/bin/ft` — execs the venv entry point with a
  **sanitized `PYTHONPATH`** (the stack env prepends onnxruntime/RCCL shims that
  must not shadow the venv's imports).
- **Registry:** `~/.mlstack/installed.json` records `freetoken` with
  `pip_packages: ["freetoken"]`.
- **Detection:** import probe under the venv python (candidates:
  `FREETOKEN_VENV_PYTHON`, `~/.mlstack/venvs/freetoken/bin/python`) — never the
  global env.

## Guard rails honored

- **No-CUDA chokepoint:** every pip invocation passes
  `execute_native_command`; the curated dep list contains no `nvidia-*`/`cuda*`/
  `+cuXXX` artifacts and the self-install is `--no-deps`, so pip can never drag
  a CUDA wheel in transitively.
- **torch provenance:** installed FIRST from `https://download.pytorch.org/whl/rocm<series>`
  (series resolved from installed ROCm) — same source as the sealed core.
- **Env respect:** `GPU_ARCH`/`HSA_OVERRIDE_GFX_VERSION` come from
  `ctx.env_exports` (sourced `~/.mlstack_env`), never hardcoded.
- **Install-time smoke test** (`VAL-INSTALL-054`): asserts `torch.version.hip`,
  auto-backend → `triton`, mapped-pinned identity mapping, and one
  `lru_ensure` slot-cache cycle — timeout-guarded (SIGALRM + GNU `timeout`).

## Verification

```bash
# status (uses the venv python + functional check: backend resolves to triton)
rusty-stack verify-enhanced   # freetoken in the enhanced set

# contamination scan: venv must contain zero nvidia-*/cuda* packages
~/.mlstack/venvs/freetoken/bin/python -m pip list | grep -iE 'nvidia|cuda'   # expect no output

# quick engine probe
~/.mlstack/bin/ft serve --model <checkpoint> --port 1919
curl http://127.0.0.1:1919/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"<checkpoint>","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
```

## Known limitations on RDNA consumer GPUs (gfx1100 class)

- fp8 / MXFP4 / NVFP4 checkpoints are **not supported** (tensor-core formats);
  use bf16 safetensors or Q4_K/Q6_K GGUF checkpoints.
- NVIDIA extras (flashinfer fused norm/sampling, sgl-kernel FA3/FA4,
  trtllm-gen) are absent by design — Triton fallbacks cover every path.
- Tensor-parallel serving via pynccl is not yet wired to RCCL (single-GPU
  serving + expert offload is the supported mode).

## Maintenance notes

- Bump the fork: push `feature/rocm`, then force-reinstall the component
  (`MLSTACK_FORCE_REINSTALL=1` or the TUI force path) — it purges the venv and
  clone, then rebuilds.
- The manifest pins `0.1.3+rocm`; update `baseline_manifest.json` when
  rebasing onto a new upstream tag.
- Upstream PR: the fork's commit series is structured for direct submission to
  `FlashML-org/FreeToken` (feature-detection style, no AMD hardcodes).
