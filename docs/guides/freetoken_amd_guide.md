# FreeToken on AMD GPUs — User Guide

Serve MoE LLMs on Radeon GPUs with experts offloaded to host RAM.

**Hardware tested:** 2× RX 7900 XTX (gfx1100), ROCm 7.2.4 · **Stack:** rusty-stack `freetoken` component

## Quick start

```bash
# 1. Install via the rusty-stack TUI (Extensions → FreeToken) or CLI:
#    (creates ~/.mlstack/venvs/freetoken, launcher at ~/.mlstack/bin/ft)

# 2. Serve the Ornith GGUF (verified: coherent chat + reasoning, ~39 tok/s decode
#    on a 7900 XTX with experts streamed from host RAM):
~/.mlstack/bin/ft serve --model /mnt/HDD-2/Models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf

# 3. Chat with it (OpenAI-compatible API on :1919):
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"ornith","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

FreeToken's core idea: MoE experts live in **pinned host RAM**, a VRAM slot
cache keeps the hot ones resident (LRU), misses stream over PCIe or compute on
CPU — so a 35B-A3B MoE serves comfortably from a 24 GB card.

## Choosing a checkpoint

| Format | RDNA3 (gfx1100) |
|---|---|
| bf16 / bf16-safetensors | ✅ |
| GGUF Q4_K / Q6_K | ✅ (qwen35moe + gemma4 archs; Ornith/Qwen3.5-class MoE verified) |
| fp8 / MXFP4 / NVFP4 | ❌ tensor-core formats, not supported |

Supported families include Qwen3.5-MoE (hybrid GDN), Qwen3-MoE, DeepSeek-V4,
GLM4-MoE, gpt-oss, MiniMax M2/M3, Llama, Gemma, Mistral. Attention runs the
engine's pure-Triton backend on AMD (auto-selected).

## Useful commands

```bash
ft serve --model <ckpt> --moe-backend hybrid   # PCIe + CPU bandwidth-adaptive split
ft bench bw                                    # memory-bandwidth benchmark
ft shell --model <ckpt>                        # interactive terminal chat
ft checkpoint --help                           # checkpoint conversion tools
```

## SSD tier + pre-warm (models larger than RAM)

Model capacity is disk-bound, not RAM-bound: with the SSD tier, expert banks
live in an NVMe pack file and the kernel page cache tiers hot/cold
automatically. Verified with the 35B Q4_K_M (16.9 GiB pack); larger models
are bounded by the NVMe mount, not host RAM.

```bash
# 1. Serve with file-backed banks (first run auto-packs, ~3.5 min from HDD):
export FREETOKEN_EXPERT_BANK_STORAGE=file
export FREETOKEN_BANK_CACHE_DIR=/mnt/WD-SSD/mlstack-banks   # your NVMe

# 2. Capture a routing profile (500+ tokens of any generation):
export FREETOKEN_EXPERT_PROFILE_OUT=$FREETOKEN_BANK_CACHE_DIR/model.profile.json

# 3. Restart with the profile to pre-warm the slot cache (decode starts warm,
#    the cold-start never faults; CUDA graphs are off in file mode):
export FREETOKEN_EXPERT_PROFILE=$FREETOKEN_BANK_CACHE_DIR/model.profile.json
```

Why pre-warm works: long-run expert usage is FLAT (no globally-hot experts)
but the moment-local set is -- a recency-ranked cache admits the right set
even though the averages say there is none. Code + chat profiles can be
merged; cross-workload profiles retain most of the benefit.

## Memory management (zram swap + the model loader)

The expert banks live in host RAM (mmap shmem). When the system zram swap
fills to 100%, the kernel OOM killer targets the loader (agent-session
processes carry `oom_score_adj=200` — the preferred victim) even with
free RAM available. Rootless fix, verified:

```bash
# 1. Find the swap holders:
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  swap=$(awk '/^VmSwap/{print $2}' /proc/$pid/status 2>/dev/null)
  [ -n "$swap" ] && [ "$swap" -gt 102400 ] && echo "$((swap/1024))MB $pid $(cat /proc/$pid/comm)"
done | sort -rn | head

# 2. Kill stale holders that auto-restart (baloo_file indexer, old firefox
#    content processes) — zram pages die with their owner, freeing headroom.
# 3. The definitive reset needs root once: sudo swapoff -a && sudo swapon -a
#    (verified safe: logical swap debt ~10GB fits in free RAM).
```

For tensor-parallel serving, the launcher keeps the RCCL overlay vars
(see the extension guide) — the stock system librccl is broken for
cross-GPU collectives.

## Troubleshooting

- **`import freetoken` fails in the global env** — by design; the package
  lives only in `~/.mlstack/venvs/freetoken`. Use the `~/.mlstack/bin/ft`
  launcher (it also sanitizes `PYTHONPATH` so stack shims like the RCCL
  overlay can't shadow venv imports).
- **First generation is slow** — tvm-ffi JIT-compiles the memory kernels via
  hipcc on first use, then caches them (`~/.cache/tvm-ffi`). Subsequent starts
  are fast.
- **`attention backend ... requires flashinfer`** — you explicitly passed
  `--attention-backend fi`/`fa`/`trtllm`; on AMD use `auto` or `triton`.
- **Out of VRAM** — lower `--moe-cache-size`, or `--moe-backend cpu`/`hybrid`
  to spill expert compute to host RAM.
